package it.unibo.alchemist.model

import it.unibo.alchemist.model.actions.AbstractLocalAction
import it.unibo.alchemist.model.implementations.actions.RunSurrogateScafiProgram
import it.unibo.alchemist.model.implementations.nodes.SimpleNodeManager
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

  private var deltaCostPerDevice: Map[Int, Double] = Map()

  override def cloneAction(node: Node[T], reaction: Reaction[T]): Action[T] = ???

  override def execute(): Unit = {
    val currentTime = environment.getSimulation.getTime.toDouble
    updateComponentCount(surrogateRunner.map(_.isSurrogateFor.size).sum)
    previousTime match {
      case None => previousTime = Some(currentTime)
      case Some(prevTime) =>
        val deltaTime = currentTime - prevTime
        previousTime = Some(currentTime)
        val cost = deltaTime * dollarsPerHourPerComponent / 3600

        deltaCostPerDevice = getLastDeltaCostPerDevice(cost)
          .map { case (id, c) => (id, deltaCostPerDevice.getOrElse(id, 0.0) + c) }

        val costLastDelta = surrogateRunner.map(_.isSurrogateFor.size).sum * cost
        totalCost += costLastDelta
        node.setConcentration(PayPerUseDevice.COST_LAST_DELTA, costLastDelta.asInstanceOf[T])
        node.setConcentration(PayPerUseDevice.TOTAL_COST, totalCost.asInstanceOf[T])
    }
  }

  def updateComponentCount(componentCount: Int): Unit = {
    if (new SimpleNodeManager[T](node).has("cloudDevice")) {
      node.setConcentration(PayPerUseDevice.COMPONENT_CLOUD, componentCount.asInstanceOf[T])
    } else if (new SimpleNodeManager[T](node).has("infrastructuralDevice")) {
      node.setConcentration(PayPerUseDevice.COMPONENT_EDGE, componentCount.asInstanceOf[T])
    }
  }

  def deltaCostPerDevice(deviceID: Int): Double = deltaCostPerDevice.get(deviceID) match {
    case Some(cost) =>
      deltaCostPerDevice = deltaCostPerDevice - deviceID
      cost
    case _ => 0.0
  }

  private def getLastDeltaCostPerDevice(deltaCost: Double): Map[Int, Double] =
    surrogateRunner
      .map(program => program.isSurrogateFor)
      .map(ids => ids.map(id => id -> deltaCost).toMap)
      .foldLeft(Map.empty[Int, Double]) { case (acc, listOfMaps) =>
        listOfMaps.foldLeft(acc) { case (map, (id, costPerComponent)) =>
          map.updated(id, map.getOrElse(id, 0.0) + costPerComponent)
        }
      }

}

private object PayPerUseDevice {
  val COST_LAST_DELTA = new SimpleMolecule("costLastDelta")
  val TOTAL_COST = new SimpleMolecule("totalCost")
  val COMPONENT_CLOUD = new SimpleMolecule("componentsInCloud")
  val COMPONENT_EDGE = new SimpleMolecule("componentsInInfrastructural")
  val IS_CLOUD = new SimpleMolecule("cloudDevice")
  val IS_EDGE = new SimpleMolecule("infrastructuralDevice")
}
