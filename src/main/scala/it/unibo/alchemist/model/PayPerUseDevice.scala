package it.unibo.alchemist.model

import it.unibo.alchemist.model.actions.AbstractLocalAction
import it.unibo.alchemist.model.implementations.actions.RunSurrogateScafiProgram
import it.unibo.alchemist.model.molecules.SimpleMolecule

import scala.jdk.CollectionConverters.CollectionHasAsScala

class PayPerUseDevice[T, P <: Position[P]](
    private val environment: Environment[T, P],
    private val node: Node[T],
    private val dollarsPerHourPerComponent: Double,
) extends AbstractLocalAction[T](node) {

  private lazy val surrogateRunner = node.getReactions.asScala
    .flatMap(_.getActions.asScala)
    .filter(_.isInstanceOf[RunSurrogateScafiProgram[T, P]])
    .map(_.asInstanceOf[RunSurrogateScafiProgram[T, P]])
    .toList

  private var previousTime: Option[Double] = None

  private var totalCost = 0.0

  override def cloneAction(node: Node[T], reaction: Reaction[T]): Action[T] = ???

  override def execute(): Unit = {
    val currentTime = environment.getSimulation.getTime.toDouble
    previousTime match {
      case None => previousTime = Some(currentTime)
      case Some(prevTime) =>
        val deltaTime = currentTime - prevTime
        previousTime = Some(currentTime)
        val cost = deltaTime * dollarsPerHourPerComponent / 3600
        val costLastDelta = surrogateRunner.map(_.isSurrogateFor.size).sum * cost
        totalCost += costLastDelta
        node.setConcentration(PayPerUseDevice.COST_LAST_DELTA, costLastDelta.asInstanceOf[T])
        node.setConcentration(PayPerUseDevice.TOTAL_COST, totalCost.asInstanceOf[T])
    }
  }
}

private object PayPerUseDevice {
  val COST_LAST_DELTA = new SimpleMolecule("costLastDelta")
  val TOTAL_COST = new SimpleMolecule("totalCost")
}