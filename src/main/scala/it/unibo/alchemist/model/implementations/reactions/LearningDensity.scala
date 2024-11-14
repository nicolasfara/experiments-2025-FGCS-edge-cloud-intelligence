package it.unibo.alchemist.model.implementations.reactions

import it.unibo.alchemist.model.{BatteryEquippedDevice, Environment, Node, PayPerUseDevice, Position, TimeDistribution}
import it.unibo.alchemist.utils.PythonModules.{rlUtils, torch}
import it.unibo.alchemist.model.molecules.SimpleMolecule
import it.unibo.alchemist.utils.AlchemistScafiUtils.getAlchemistActions
import it.unibo.alchemist.utils.Molecules
import me.shadaj.scalapy.py.SeqConverters
import me.shadaj.scalapy.py

import scala.jdk.CollectionConverters.CollectionHasAsScala

class LearningDensity[T, P <: Position[P]](
    environment: Environment[T, P],
    distribution: TimeDistribution[T],
    seed: Int,
) extends GraphBuilderReaction[T, P](environment, distribution) {

  private val rewardFunction = rlUtils.DensityRewardFunction()

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

      var densityLevel = 0
      if (node.contains(new SimpleMolecule("distance"))) {
        val distance = node.getConcentration(new SimpleMolecule("distance")).asInstanceOf[Boolean]
        if (distance) {
          densityLevel = 1
        }
      }

//      println(s"[DEBUG] local components $localComponents")

      val latencies: Seq[Double] = componentsAllocation.map {
        case (componentId, where) if where == node.getId =>
          0.0
        case (_, where) =>
          val density = node.getConcentration(new SimpleMolecule(Molecules.density)).asInstanceOf[Double]
          density match {
            case d if d < 5.0 => 0.0
            case d if d < 15  => 0.2
            case _            => 2.0
          }
//            baseLatency(where) *  density
      }.toSeq

      val f = Seq(batteryLevel, edgeServerDeltaCost, cloudDeltaCost, localComponents) ++ latencies
      Vector(f)
    } else {
      val cost = node.getConcentration(PayPerUseDevice.TOTAL_COST).asInstanceOf[Double]
      Vector(Seq(cost))
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
    applicationNodes
      .map(_.getId)
      .contains(neigh.getId) match {
      case applicationNode if applicationNode =>
        val distance = environment.getPosition(node).distanceTo(environment.getPosition(neigh))
        Vector(Seq(distance))
      case _ =>
        val density = node.getConcentration(new SimpleMolecule(Molecules.density)).asInstanceOf[Double]
        val latency = baseLatency(neigh.getId) * density
        Vector(Seq(latency))
    }
  }

  private def baseLatency(nodeId: Int): Double = {
    nodeId match {
      case id if infrastructuralNodes.map(_.getId).contains(id) => 10
      case _                                                    => 100
    }
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
    val rewards = rewardFunction.compute(obs, nextObs)
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
