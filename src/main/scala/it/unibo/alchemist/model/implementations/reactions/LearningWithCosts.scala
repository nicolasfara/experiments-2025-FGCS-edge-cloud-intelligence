package it.unibo.alchemist.model.implementations.reactions

import it.unibo.alchemist.model.{Environment, Node, PayPerUseDevice, Position, TimeDistribution}
import it.unibo.alchemist.utils.PythonModules.{rlUtils, torch}
import it.unibo.alchemist.model.molecules.SimpleMolecule
import it.unibo.alchemist.utils.AlchemistScafiUtils.getAlchemistActions
import it.unibo.alchemist.utils.Molecules
import me.shadaj.scalapy.py.SeqConverters
import me.shadaj.scalapy.py

class LearningWithCosts[T, P <: Position[P]](
    environment: Environment[T, P],
    distribution: TimeDistribution[T],
    seed: Int,
) extends GraphBuilderReaction[T, P](environment, distribution) {

  private val rewardFunction = rlUtils.CostRewardFunction()

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

      val f = Seq(edgeServerDeltaCost, cloudDeltaCost, localComponents)

      Vector(f)
    } else {
      val cost = node.getConcentration(PayPerUseDevice.TOTAL_COST).asInstanceOf[Double]
      Vector(Seq(cost))
    }
  }

  private def getDeltaCost(nodes: Seq[Node[T]], mid: Int): Double=
    nodes
      .map(_.getId)
      .filter(remoteID => nodes.map(_.getId).contains(remoteID))
      .map (remoteID => getAlchemistActions(environment, remoteID, classOf[PayPerUseDevice[T, P]]))
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

    val localComponents = newAllocation.values.count(_ == node.getId).toDouble
    val localComponentsPercentage = localComponents / components.size.toDouble
    node.setConcentration(
      new SimpleMolecule(Molecules.localComponentsPercentage),
      localComponentsPercentage.asInstanceOf[T],
    )
  }

  override protected def computeRewards(obs: py.Dynamic, nextObs: py.Dynamic): py.Dynamic = {
    val rewards = rewardFunction.compute(obs, nextObs)
//    println(environment.getSimulation.getTime.toDouble)
    //    println(s"[DEBUG] Rewards: $rewards")

    rewards
      .tolist()
      .as[List[Double]]
      .zipWithIndex
      .foreach { case (reward, index) =>
        applicationNodes(index).setConcentration(new SimpleMolecule(Molecules.reward), reward.asInstanceOf[T])
      }
//    torch.Tensor(rewards.toPythonProxy)
    rewards
  }

  override protected def getSeed: Int = seed
}
