import scala.annotation.tailrec
import scala.util.Random.{nextLong, nextFloat}
import scala.collection.immutable.ArraySeq
import scala.math.Ordering.Float.TotalOrdering
import scala.concurrent.{ExecutionContext, Future, Await}
import scala.concurrent.duration.Duration

import java.util.concurrent.Executors
import java.awt.image.{BufferedImage, DataBufferUShort}
import java.io.File
import javax.imageio.ImageIO

import Math.{min, max, hypot, floorMod, sin, cos, PI, exp}
import System.currentTimeMillis

val numCpus = Runtime.getRuntime.nn.availableProcessors()

def main(args: Array[String]): Unit =
  // Run the erosion algorithm on a heightmap initialized with fractal noise.
  // Save the initial and resulting heightmaps to PNG files.

  val sizeX = 1024
  val sizeY = 1024
  var heightMap = Array.fill(sizeX * sizeY)(0f)

  val es = Executors.newFixedThreadPool(numCpus).nn
  implicit val ec: ExecutionContext = ExecutionContext.fromExecutorService(es)

  val noise = OpenSimplexNoise(nextLong())

  for x <- 0 until sizeX; y <- 0 until sizeY do
    val n = fractalize(x * 0.003f, y * 0.003f, noise.eval(_, _).toFloat, 10)
    heightMap(x + y * sizeX) = (n.toFloat + 1) / 2

  val image = BufferedImage(sizeX, sizeY, BufferedImage.TYPE_USHORT_GRAY)
  val imageData = image.getRaster.nn.getDataBuffer.asInstanceOf[DataBufferUShort].getData.nn

  def copyHeightMap(): Unit =
    val minHeight = heightMap.min
    val maxHeight = heightMap.max
    for i <- 0 until sizeX * sizeY do
      imageData(i) = ((heightMap(i) - minHeight) / maxHeight * 0xffff).round.toShort

  copyHeightMap()

  val beforeFile = File("before.png")
  beforeFile.createNewFile()
  ImageIO.write(image, "png", beforeFile)

  val startTime = currentTimeMillis()

  heightMap = erodeHeightMap(
    heightMap,
    sizeX = sizeX,
    sizeY = sizeY,
    iterations = 512,
    cellSize = 200f / max(sizeX, sizeY),
    rainRate = 0.0004f,
    evaporationRate = 0.0005f,
    minHeightDelta = 0.05f,
    gravity = 30f,
    dissolveRate = 0.25f,
    depositionRate = 0.001f,
    sedimentCapCoeff = 50f,
    reposeSlope = 0.05f,
  )

  val endTime = currentTimeMillis()

  copyHeightMap()

  val afterFile = File("after.png")
  afterFile.createNewFile()
  ImageIO.write(image, "png", afterFile)

  println(f"Erosion took ${(endTime - startTime) / 1000.0}%.2fs")
  assert(es.shutdownNow().nn.size == 0)

def fractalize(
  x: Float,
  y: Float,
  f: (Float, Float) => Float,
  octaves: Int = 1,
  lacunarity: Float = 2f,
  persistence: Float = 0.5f,
): Double =
  @tailrec
  def loop(o: Int, l: Float, p: Float, d: Float, s: Float): Float =
    if o < octaves then
      loop(o + 1, l * lacunarity, p * persistence, d + p, s + f(x * l, y * l) * p)
    else s / d
  loop(0, 1, 1, 0, 0)

def lerp(a: Float, b: Float, t: Float): Float =
  a * (1 - t) + b * t

def bilerp(v00: Float, v10: Float, v01: Float, v11: Float, tx: Float, ty: Float): Float =
  lerp(lerp(v00, v10, tx), lerp(v01, v11, tx), ty)

def gaussBlurFn(x: Float, y: Float, sigma: Float): Float =
  val sigmaSq2 = sigma * sigma * 2
  1 / (PI.toFloat * sigmaSq2) * exp(-(x * x + y * y) / sigmaSq2).toFloat

// The actual erosion code.
def erodeHeightMap(
  heightMap: Array[Float],
  sizeX: Int,
  sizeY: Int,
  iterations: Int,
  cellSize: Float,
  rainRate: Float,
  evaporationRate: Float,
  minHeightDelta: Float,
  gravity: Float,
  dissolveRate: Float,
  depositionRate: Float,
  sedimentCapCoeff: Float,
  reposeSlope: Float,
)(implicit ec: ExecutionContext): Array[Float] =

  def get(a: Array[Float], x: Int, y: Int): Float =
    a(floorMod(x, sizeX) + floorMod(y, sizeY) * sizeX)

  def set(a: Array[Float], x: Int, y: Int, v: Float): Unit =
    a(floorMod(x, sizeX) + floorMod(y, sizeY) * sizeX) = v

  def parallel2d(f: (Int, Int) => Unit): Unit =
    val chunk = sizeX * sizeY / numCpus
    val fut = Future.sequence(
      for i <- 1 to numCpus yield Future {
        for j <- chunk * (i - 1) until chunk * i do
          f(j % sizeX, j / sizeX)
      }
    )
    for i <- chunk * numCpus until sizeX * sizeY do
      f(i % sizeX, i / sizeX)
    Await.ready(fut, Duration.Inf)

  var terrain = heightMap
  var terrain2 = Array.fill(sizeX * sizeY)(0f)
  var water = terrain2.clone
  var water2 = terrain2.clone
  var sediment = terrain2.clone
  var sediment2 = terrain2.clone
  val speed = terrain2.clone
  val gradientX = terrain2.clone
  val gradientY = terrain2.clone

  val weightMatrixSize = 5
  val blurWeightMatrix =
    val matrix =
      for y <- 0 until weightMatrixSize; x <- 0 until weightMatrixSize
      yield gaussBlurFn(
        (x - weightMatrixSize / 2).toFloat,
        (y - weightMatrixSize / 2).toFloat,
        1
      )
    val sum = matrix.sum
    ArraySeq.from(matrix.map(_ / sum))

  // Gets a value in the array at the given coords using bilinear interpolation.
  def sample(a: Array[Float], x: Float, y: Float): Float =
    val ix = x.floor.toInt
    val iy = y.floor.toInt
    bilerp(
      get(a, ix, iy),
      get(a, ix + 1, iy),
      get(a, ix, iy + 1),
      get(a, ix + 1, iy + 1),
      x - ix,
      y - iy
    )

  for iter <- 1 to iterations do
    parallel2d { (x, y) =>
      set(water, x, y, get(water, x, y) + rainRate * cellSize * cellSize)

      val (gradX, gradY) =
        val dx = get(terrain, x - 1, y) - get(terrain, x + 1, y)
        val dy = get(terrain, x, y - 1) - get(terrain, x, y + 1)
        val mag = hypot(dx, dy).toFloat
        if mag < 1e-10 then
          val angle = nextFloat() * PI * 2
          (cos(angle).toFloat, sin(angle).toFloat)
        else (dx / mag, dy / mag)

      set(gradientX, x, y, gradX)
      set(gradientY, x, y, gradY)

      val oldHeight = get(terrain, x, y)
      val newHeight = sample(terrain, x + gradX, y + gradY)
      val heightDelta = oldHeight - newHeight

      val sedimentCap =
        max(heightDelta, minHeightDelta) / cellSize *
        get(speed, x, y) *
        get(water, x, y) *
        sedimentCapCoeff

      val depositedSediment =
        val sed = get(sediment, x, y)
        max(
          -heightDelta,
          if heightDelta < 0 then min(heightDelta, sed)
          else if sed > sedimentCap then depositionRate * (sed - sedimentCap)
          else dissolveRate * (sed - sedimentCap)
        )

      set(sediment, x, y, get(sediment, x, y) - depositedSediment)
      set(terrain2, x, y, get(terrain, x, y) + depositedSediment)
      set(speed, x, y, gravity * heightDelta / cellSize)
    }

    // Transport water and sediment in the Moore neighborhood using the normalized gradient vectors.
    parallel2d { (x, y) =>
      val wrd = max(-get(gradientX, x + 1, y + 1), 0f) * max(-get(gradientY, x + 1, y + 1), 0f)
      val wrm = max(-get(gradientX, x + 1, y), 0f) * max(1 - get(gradientY, x + 1, y).abs, 0f)
      val wru = max(-get(gradientX, x + 1, y - 1), 0f) * max(get(gradientY, x + 1, y - 1), 0f)
      val wmd = max(1 - get(gradientX, x, y + 1).abs, 0f) * max(-get(gradientY, x, y + 1), 0f)
      val wmm = max(1 - get(gradientX, x, y).abs, 0f) * max(1 - get(gradientY, x, y).abs, 0f)
      val wmu = max(1 - get(gradientX, x, y - 1).abs, 0f) * max(get(gradientY, x, y - 1), 0f)
      val wld = max(get(gradientX, x - 1, y + 1), 0f) * max(-get(gradientY, x - 1, y + 1), 0f)
      val wlm = max(get(gradientX, x - 1, y), 0f) * max(1 - get(gradientY, x - 1, y).abs, 0f)
      val wlu = max(get(gradientX, x - 1, y - 1), 0f) * max(get(gradientY, x - 1, y - 1), 0f)
      
      set(sediment2, x, y, 
        get(sediment, x + 1, y + 1) * wrd +
        get(sediment, x + 1, y) * wrm +
        get(sediment, x + 1, y - 1) * wru +
        get(sediment, x, y + 1) * wmd +
        get(sediment, x, y) * wmm +
        get(sediment, x, y - 1) * wmu +
        get(sediment, x - 1, y + 1) * wld +
        get(sediment, x - 1, y) * wlm +
        get(sediment, x - 1, y - 1) * wlu
      )

      set(water2, x, y,
        get(water, x + 1, y + 1) * wrd +
        get(water, x + 1, y) * wrm +
        get(water, x + 1, y - 1) * wru +
        get(water, x, y + 1) * wmd +
        get(water, x, y) * wmm +
        get(water, x, y - 1) * wmu +
        get(water, x - 1, y + 1) * wld +
        get(water, x - 1, y) * wlm +
        get(water, x - 1, y - 1) * wlu
      )
    }

    // Smooth out slopes that are too steep using gaussian blur.
    parallel2d { (x, y) =>
      // Evaporate the water while we're at it.
      set(water2, x, y, get(water2, x, y) * (1 - evaporationRate))

      val slope = hypot(
        (get(terrain2, x + 1, y) - get(terrain2, x - 1, y)) / 2 / cellSize,
        (get(terrain2, x, y + 1) - get(terrain2, x, y - 1)) / 2 / cellSize
      )
      if slope > reposeSlope then
        var sum = 0f
        for wx <- 0 until weightMatrixSize do
          for wy <- 0 until weightMatrixSize do
            sum +=
              blurWeightMatrix(wx + wy * weightMatrixSize) *
              get(terrain2, x + wx - weightMatrixSize / 2, y + wy - weightMatrixSize / 2)
        set(terrain, x, y, sum)
      else
        set(terrain, x, y, get(terrain2, x, y))
    }

    var tmp = water
    water = water2
    water2 = tmp

    tmp = sediment
    sediment = sediment2
    sediment2 = tmp

    println(s"$iter / $iterations")

  terrain
