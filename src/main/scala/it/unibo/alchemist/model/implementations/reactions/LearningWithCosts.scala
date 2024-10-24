package it.unibo.alchemist.model.implementations.reactions

import it.unibo.alchemist.model.{Environment, Node, PayPerUseDevice, Position, TimeDistribution}
import it.unibo.alchemist.utils.PythonModules.{rlUtils, torch}
import it.unibo.alchemist.model.molecules.SimpleMolecule
import it.unibo.alchemist.utils.Molecules
import me.shadaj.scalapy.py.SeqConverters
import me.shadaj.scalapy.py

class LearningWithCosts[T, P <: Position[P]](
    environment: Environment[T, P],
    distribution: TimeDistribution[T],
    seed: Int
) extends GraphBuilderReaction[T, P](environment, distribution) {

  private val rewardFunction = rlUtils.CostRewardFunction()

  override protected def getNodeFeature(node: Node[T]): Vector = {
    if(!node.contains(new SimpleMolecule(Molecules.infrastructural))) {
      Vector(Seq())
    }
    else {
      val cost = node.getConcentration(PayPerUseDevice.COST_LAST_DELTA).asInstanceOf[Double]
      Vector(Seq(cost))
    }
  }

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
      localComponentsPercentage.asInstanceOf[T]
    )
  }

  override protected def computeRewards(obs: py.Dynamic, nextObs: py.Dynamic): py.Dynamic = {
    val rewards = rewardFunction.compute(obs, nextObs).tolist().as[List[Double]]
    rewards
      .zipWithIndex
      .foreach { case (reward, index) =>
        applicationNodes(index).setConcentration(new SimpleMolecule(Molecules.reward), reward.asInstanceOf[T])
      }
    torch.Tensor(rewards.toPythonProxy)
  }

  override protected def getSeed: Int = seed
}
