package it.unibo.alchemist

class GreaterDistance extends MyAggregateProgram {

  override def main(): Any = {
    val distanceFromSource = inputFromComponent("it.unibo.alchemist.Gradient", Double.PositiveInfinity)
    val result = distanceFromSource > 10.0
    writeEnv("distance", result)
    result
  }
}
