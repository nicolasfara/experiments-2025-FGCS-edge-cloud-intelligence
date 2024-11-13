package it.unibo.alchemist

import it.unibo.alchemist.model.scafi.ScafiIncarnationForAlchemist._

class DensityEstimation extends MyAggregateProgram with FieldUtils with ExplicitFields {

  override def main(): Double = localDensity()

  private def localDensity(): Double =  foldhoodPlus(0.0)(_ + _)(1 / nbrRange())
}
