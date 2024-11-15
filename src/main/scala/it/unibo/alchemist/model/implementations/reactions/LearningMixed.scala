package it.unibo.alchemist.model.implementations.reactions

import it.unibo.alchemist.model.{BatteryEquippedDevice, Environment, Node, PayPerUseDevice, Position, TimeDistribution}
import it.unibo.alchemist.utils.PythonModules.{rlUtils, torch}
import it.unibo.alchemist.model.molecules.SimpleMolecule
import it.unibo.alchemist.utils.AlchemistScafiUtils.getAlchemistActions
import it.unibo.alchemist.utils.Molecules
import me.shadaj.scalapy.py.SeqConverters
import me.shadaj.scalapy.py

import scala.jdk.CollectionConverters.CollectionHasAsScala

class LearningMixed[T, P <: Position[P]](
    environment: Environment[T, P],
    distribution: TimeDistribution[T],
    seed: Int,
    alpha: Double,
    beta: Double,
    gamma: Double,
) extends GraphBuilderReaction[T, P](environment, distribution) {

  private val rewardFunction = rlUtils.MixedRewardFunction()

  override protected def getNodeFeature(node: Node[T]): Vector = {
    if (!node.contains(new SimpleMolecule(Molecules.infrastructural)) && !node.contains(new SimpleMolecule(Molecules.cloud))) {
      val componentsAllocation = getAllocator(node).getComponentsAllocation
      val totalComponents = componentsAllocation.size
      val localComponents = componentsAllocation.count { case (_, where) => node.getId == where }
      val edgeServerComponents = componentsAllocation
        .count { case (_, where) => infrastructuralNodes.map(_.getId).contains(where) }
      val cloudComponents = componentsAllocation
        .count { case (_, where) => cloudNodes.map(_.getId).contains(where) }
      val edgeServerDeltaCost = getDeltaCost(infrastructuralNodes, node.getId)
      val cloudDeltaCost = getDeltaCost(cloudNodes, node.getId)
      val batteryLevel = BatteryEquippedDevice.getBatteryPercentage(node)
      val allocated = componentsAllocation
        .filterNot(_._2 == node.getId)
        .filterNot(a => cloudNodes.map(_.getId).contains(a._2))
        .map { case (_, id) => getAlchemistActions(environment, id, classOf[PayPerUseDevice[T, P]]) }
        .headOption match {
        case Some(device) =>
          device
            .map(_.getComponentsCount.toDouble)
            .toSeq
            .appendedAll(Seq.fill(componentsAllocation.size)(0.0))
            .take(componentsAllocation.size)
        case None => Seq.fill(componentsAllocation.size)(0.0)
      }
      val f = Seq(batteryLevel, edgeServerDeltaCost, cloudDeltaCost, localComponents) ++ allocated
      Vector(f)
    } else {
      val cost = node.getConcentration(PayPerUseDevice.TOTAL_COST).asInstanceOf[Double]
      val components = node.getConcentration(PayPerUseDevice.COMPONENTS).asInstanceOf[Double]
      val costPerHour = node.getConcentration(PayPerUseDevice.COST_PER_HOUR).asInstanceOf[Double]
      Vector(Seq(cost, components, costPerHour))
    }
  }

  private def getDeltaCost(nodes: Seq[Node[T]], mid: Int): Double =
    nodes
      .map(_.getId)
      .filter(remoteID => nodes.map(_.getId).contains(remoteID))
      .map(remoteID => getAlchemistActions(environment, remoteID, classOf[PayPerUseDevice[T, P]]))
      .map(_.head)
      .map(_.deltaCostPerDevice(mid))
      .sum

  override protected def getEdgeFeature(node: Node[T], neigh: Node[T]): Vector = {
    val distance = environment.getPosition(node).distanceTo(environment.getPosition(neigh))
    Vector(Seq(distance))
  }

  override protected def updateAllocation(node: Node[T], newAllocation: Map[String, Int]): Unit = {
    getAllocator(node)
      .setComponentsAllocation(newAllocation)

    val batteryModel = node.getReactions.asScala
      .flatMap(_.getActions.asScala)
      .find(_.isInstanceOf[BatteryEquippedDevice[T, P]])
      .map(_.asInstanceOf[BatteryEquippedDevice[T, P]])
      .getOrElse(throw new IllegalStateException("Battery action not found!"))

    batteryModel.updateComponentsExecution(newAllocation)

    val localComponents = newAllocation.values.count(_ == node.getId).toDouble
    val localComponentsPercentage = localComponents / components.size.toDouble
    node.setConcentration(
      new SimpleMolecule(Molecules.localComponentsPercentage),
      localComponentsPercentage.asInstanceOf[T],
    )
  }

  override protected def computeRewards(obs: py.Dynamic, nextObs: py.Dynamic): py.Dynamic = {
    val rewards = rewardFunction.compute(obs, nextObs, alpha, beta, gamma)
    rewards
      .tolist()
      .as[List[Double]]
      .zipWithIndex
      .foreach { case (reward, index) =>
        applicationNodes(index).setConcentration(new SimpleMolecule(Molecules.reward), reward.asInstanceOf[T])
      }
    rewards
  }

  override protected def getSeed: Int = seed
}
