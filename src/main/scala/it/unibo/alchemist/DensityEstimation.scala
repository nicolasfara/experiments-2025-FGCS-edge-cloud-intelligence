package it.unibo.alchemist

import it.unibo.alchemist.model.scafi.ScafiIncarnationForAlchemist._

class DensityEstimation extends MyAggregateProgram with FieldUtils with ExplicitFields {

  override def main(): Double = {
    val (densitiesSum, distancesSum) = foldhoodPlus((0.0, 0.0)) {
      case ((sum, weightSum), (neighborDensity, neighborDistance)) =>
        (sum + neighborDensity * neighborDistance, weightSum + neighborDistance)
    }((nbr(localDensity()), nbrRange()))
    densitiesSum / distancesSum / 2.5
  }

  private def localDensity(): Double =  foldhoodPlus(0.0)(_ + _)(1 / nbrRange())
}
