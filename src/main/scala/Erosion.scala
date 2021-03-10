import scala.annotation.tailrec
import scala.util.Random.{nextDouble, nextFloat, nextLong}
import scala.collection.immutable.ArraySeq
import scala.math.Ordering.Float.TotalOrdering

import java.awt.image.{BufferedImage, DataBufferUShort}
import java.io.File
import javax.imageio.ImageIO

import Math.{min, max, hypot, floorMod}
import System.currentTimeMillis

def main(args: Array[String]): Unit =
  val sizeX = 500
  val sizeY = 500

  var heightMap = Array.fill(sizeX * sizeY)(0f)

  val noise = OpenSimplexNoise(nextLong())

  // Initialize the heightmap.
  for x <- 0 until sizeX; y <- 0 until sizeY do
    //val h = 1 - smoothstep(hypot(x - sizeX / 2, y - sizeY / 2), 0, min(sizeX, sizeY) / 4)
    //heightMap(x + y * sizeX) = (h * 0.9 + 0.1).toFloat
    val n = fractalize(x * 0.01, y * 0.01, noise.eval(_, _), 10)
    heightMap(x + y * sizeX) = (n.toFloat + 1) / 2

  val image = BufferedImage(sizeX, sizeY, BufferedImage.TYPE_USHORT_GRAY)
  val imageData = image.getRaster.getDataBuffer.asInstanceOf[DataBufferUShort].getData.nn

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

  heightMap = hydraulicErosion(
    heightMap,
    sizeX,
    sizeY,
    iterations = 100,
    rainRate = 0.01,
    soluability = 0.01,
    evaporation = 0.5,
    sedimentCapCoeff = 0.01
  )

  val endTime = currentTimeMillis()

  copyHeightMap()

  val afterFile = File("after.png")
  afterFile.createNewFile()
  ImageIO.write(image, "png", afterFile)

  println(f"Erosion took ${(endTime - startTime) / 1000.0}%.2fs")

def smoothstep(x: Double, edge0: Double, edge1: Double): Double =
  val n = clamp((x - edge0) / (edge1 - edge0), 0, 1)
  n * n * (3 - 2 * n)

def clamp(x: Double, min: Double, max: Double): Double =
  if x < min then min
  else if x > max then max
  else x

def fractalize(
  x: Double,
  y: Double,
  f: (Double, Double) => Double,
  octaves: Int = 1,
  lacunarity: Double = 2,
  persistence: Double = 0.5,
): Double =
  @tailrec
  def loop(o: Int, l: Double, p: Double, d: Double, s: Double): Double =
    if o < octaves then
      loop(o + 1, l * lacunarity, p * persistence, d + p, s + f(x * l, y * l) * p)
    else s / d
  loop(0, 1, 1, 0, 0)

def b2i(b: Boolean): 0 | 1 = if b then 1 else 0

// Hashing stuff for randomized precipitation amounts.
def hash(x: Long): Long =
  val x1 = (x ^ (x >>> 30)) * 0xbf58476d1ce4e5b9L
  val x2 = (x1 ^ (x1 >>> 27)) * 0x94d049bb133111ebL
  x2 ^ (x2 >>> 31)

def hashCombine(l: Long, r: Long): Long =
  l ^ (r + 0x9e3779b97f4a7c16L + (l << 6) + (l >>> 2))

def hydraulicErosion(
  heightMap: Array[Float],
  sizeX: Int,
  sizeY: Int,
  iterations: Int,
  rainRate: Float,
  soluability: Float,
  evaporation: Float,
  sedimentCapCoeff: Float
): Array[Float] =

  var terrain = heightMap
  var terrainAfter = Array.fill(sizeX * sizeY)(0f)
  var water = terrainAfter.clone
  var waterAfter = terrainAfter.clone
  var sediment = terrainAfter.clone
  var sedimentAfter = terrainAfter.clone

  def get(a: Array[Float], x: Int, y: Int): Float =
    a(floorMod(x, sizeX) + floorMod(y, sizeY) * sizeX)

  def set(a: Array[Float], x: Int, y: Int, v: Float): Unit =
    a(floorMod(x, sizeX) + floorMod(y, sizeY) * sizeX) = max(0, v)

  case class Outflow(
    w0: Float,
    s0: Float,
    w1: Float,
    s1: Float,
    w2: Float,
    s2: Float,
    w3: Float,
    s3: Float
  )

  // Determine the water and sediment outflow from a cell in the cardinal
  // directions at the current time.
  def cellOutflow(x: Int, y: Int): Outflow =
    val h = get(terrain, x, y)
    val w = get(water, x, y)
    val s = get(sediment, x, y)

    // Total altutude for a cell is the sum of its terrain height and water.
    val a = h + w

    val a0 = get(terrain, x, y + 1) + get(water, x, y + 1)
    val a1 = get(terrain, x - 1, y) + get(water, x - 1, y)
    val a2 = get(terrain, x + 1, y) + get(water, x + 1, y)
    val a3 = get(terrain, x, y - 1) + get(water, x, y - 1)

    // Only account for the neighboring cells that have an altitude less than
    // the current cell altitude.
    val aAvg =
      (a + a0 * b2i(a > a0) + a1 * b2i(a > a1) + a2 * b2i(a > a2) + a3 * b2i(a > a3)) /
      (1 + b2i(a > a0) + b2i(a > a1) + b2i(a > a2) + b2i(a > a3))

    val d0 = max(0, aAvg - a0)
    val d1 = max(0, aAvg - a1)
    val d2 = max(0, aAvg - a2)
    val d3 = max(0, aAvg - a3)

    val dTotal = d0 + d1 + d2 + d3

    val da = max(0, a - aAvg)

    val (dw0, dw1, dw2, dw3) =
      if dTotal == 0 then (0f, 0f, 0f, 0f) else
        (
          min(w, da) * (d0 / dTotal),
          min(w, da) * (d1 / dTotal),
          min(w, da) * (d2 / dTotal),
          min(w, da) * (d3 / dTotal),
        )

    val ds0 = s * (dw0 / w)
    val ds1 = s * (dw1 / w)
    val ds2 = s * (dw2 / w)
    val ds3 = s * (dw3 / w)

    assert(dw0 + dw1 + dw2 + dw3 < w + 0.00001, dw0 + dw1 + dw2 + dw3)
    assert(ds0 + ds1 + ds2 + ds3 < s + 0.00001, ds0 + ds1 + ds2 + ds3)

    Outflow(dw0, ds0, dw1, ds1, dw2, ds2, dw3, ds3)

  for iter <- 0 until iterations do

    // TODO: do this loop in parallel.
    for x <- 0 until sizeX; y <- 0 until sizeY do
      // Add water from precipitation and convert terrain into sediment.

      //val hashPos = hashCombine(hash(floorMod(x, sizeX) + floorMod(y, sizeY) * sizeX), hash(iter))
      //val w = get(water, x, y) + 0.00001f + hashPos.abs.toFloat / Long.MaxValue * rainRate
      val w = get(water, x, y) + rainRate
      assert(w > 0)
      set(terrain, x, y, get(terrain, x, y) - w * soluability)
      set(sediment, x, y, get(sediment, x, y) + w * soluability)
      set(water, x, y, w)

    // TODO: do this loop in parallel.
    for x <- 0 until sizeX; y <- 0 until sizeY do
      val o = cellOutflow(x, y)
      val o0 = cellOutflow(x, y + 1)
      val o1 = cellOutflow(x - 1, y)
      val o2 = cellOutflow(x + 1, y)
      val o3 = cellOutflow(x, y - 1)
      
      // The amount of water in this cell is the current amount of water minus
      // the amount of water leaving the cell plus the amount of water entering
      // the cell. Then, some of the water is evaporated.
      val w =
        (get(water, x, y) - o.w0 - o.w1 - o.w2 - o.w3 + o0.w3 + o1.w2 + o2.w1 + o3.w0) * (1 - evaporation)
      // Likewise for the sediment, but without the evaporation.
      val s =
        get(sediment, x, y) - o.s0 - o.s1 - o.s2 - o.s3 + o0.s3 + o1.s2 + o2.s1 + o3.s0

      // Add dissolved sediment back into the terrain if there is more sediment
      // in the water than the water can carry.
      val ds = max(0, s - (sedimentCapCoeff * w))
      set(waterAfter, x, y, w)
      set(sedimentAfter, x, y, s - ds)
      set(terrainAfter, x, y, get(terrain, x, y) + ds)

    // Swap terrain, water, and sediment buffers.
    var tmp = terrain
    terrain = terrainAfter
    terrainAfter = tmp
    tmp = water
    water = waterAfter
    waterAfter = tmp
    tmp = sediment
    sediment = sedimentAfter
    sedimentAfter = tmp

  terrain
