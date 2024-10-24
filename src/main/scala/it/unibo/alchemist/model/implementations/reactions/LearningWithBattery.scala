package it.unibo.alchemist.model.implementations.reactions

import it.unibo.alchemist.model.molecules.SimpleMolecule
import it.unibo.alchemist.model._
import it.unibo.alchemist.utils.Molecules
import it.unibo.alchemist.utils.PythonModules.{rlUtils, torch}
import learning.model.{ActionSpace, Component, EdgeServer, MySelf, PairComponentDevice, ExponentialDecay}
import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.SeqConverters

import scala.jdk.CollectionConverters.CollectionHasAsScala
import scala.language.implicitConversions

class LearningWithBattery[T, P <: Position[P]](
    environment: Environment[T, P],
    distribution: TimeDistribution[T],
    seed: Int
) extends GraphBuilderReaction[T, P](environment, distribution) {

  private val rewardFunction = rlUtils.BatteryRewardFunction()

  override protected def getNodeFeature(node: Node[T]): Vector = {
    if(!node.contains(new SimpleMolecule(Molecules.infrastructural))) {
      val batteryLevel = BatteryEquippedDevice.getBatteryPercentage(node)
      val componentsAllocation = getAllocator(node)
        .getComponentsAllocation
      val totalComponents = componentsAllocation.size
      val localComponents = componentsAllocation.count { case (_, where) => node.getId == where }
      Vector(Seq(batteryLevel, (localComponents / totalComponents).toDouble))
    }
    else {
      Vector(Seq())
    }
  }

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
    node.setConcentration(new SimpleMolecule(Molecules.localComponentsPercentage), localComponentsPercentage.asInstanceOf[T])
  }

  override protected def computeRewards(obs: py.Dynamic, nextObs: py.Dynamic): py.Dynamic = {
    val rewards = rewardFunction.compute_difference(obs, nextObs).tolist().as[List[Double]]
    rewards
      .zipWithIndex
      .foreach { case (reward, index) =>
        applicationNodes(index).setConcentration(new SimpleMolecule(Molecules.reward), reward.asInstanceOf[T])
      }
    torch.Tensor(rewards.toPythonProxy)
  }

  override protected def getSeed: Int = seed

}