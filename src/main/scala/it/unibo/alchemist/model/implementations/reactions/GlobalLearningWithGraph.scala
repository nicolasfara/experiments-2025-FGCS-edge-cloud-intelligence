package it.unibo.alchemist.model.implementations.reactions

import it.unibo.alchemist.model.molecules.SimpleMolecule
import it.unibo.alchemist.model._
import it.unibo.alchemist.utils.Molecules
import it.unibo.alchemist.utils.PythonModules.{rlUtils, torch}
import learning.model.{ActionSpace, Component, EdgeServer, MySelf, PairComponentDevice}
import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.SeqConverters

import scala.jdk.CollectionConverters.CollectionHasAsScala
import scala.language.implicitConversions

class GlobalLearningWithGraph[T, P <: Position[P]](
    environment: Environment[T, P],
    distribution: TimeDistribution[T],
    seed: Int
) extends GraphBuilderReaction[T, P](environment, distribution) {

  private lazy val edgeServerSize = infrastructuralNodes.size
  private lazy val components = getComponents

  private var oldGraph: Option[py.Dynamic] = None
  private var oldActions: Option[py.Dynamic] = None
  private val rewardFunction = rlUtils.BatteryRewardFunction()
  private lazy val actionSpace = ActionSpace(components, edgeServerSize)

  override protected def getNodeFeature(node: Node[T]): Vector = {
    val batteryLevel = BatteryEquippedDevice.getBatteryCapacity(node)
    val componentsAllocation = getAllocator(node)
      .getComponentsAllocation
    val totalComponents = componentsAllocation.size
    val localComponents = componentsAllocation.count { case (_, where) => node.getId == where }
    Vector(Seq(batteryLevel, (localComponents / totalComponents).toDouble))
  }

  override protected def getEdgeFeature(node: Node[T], neigh: Node[T]): Vector = {
    val distance = environment.getPosition(node).distanceTo(environment.getPosition(neigh))
    Vector(Seq(distance))
  }

  override protected def handleGraph(observation: py.Dynamic): Unit = {

    val actions = learner.select_action(observation, 0.05) // TODO inject epsilon from outside

    actions
      .tolist().as[List[Int]]
      .zipWithIndex
      .foreach { case (actionIndex, nodeIndex) =>
        val node = applicationNodes(nodeIndex)
        val newComponentsAllocation = actionSpace.actions(actionIndex)
          .map { case PairComponentDevice(component, device) =>
            val deviceID = device match {
              case MySelf() => node.getId
              case EdgeServer(id) => id
            }
            component.id -> deviceID
          }
          .toMap
        getAllocator(node)
          .setComponentsAllocation(newComponentsAllocation)
      }

    (oldGraph, oldActions) match {
      case (Some(previousObs), Some(previousActions)) =>
        val rewards = computeRewards(previousObs, observation)
        learner.add_experience(previousObs, previousActions, rewards, observation)
        learner.train_step_dqn(batch_size=32, gamma=0.99, update_target_every=10, seed=seed)
      case _ =>
    }

    oldGraph = Some(observation)
    oldActions = Some(actions)
  }

  private def computeRewards(obs: py.Dynamic, nextObs: py.Dynamic): py.Dynamic = {
    val rewards = rewardFunction.compute_threshold(obs, nextObs).tolist().as[List[Int]]
    rewards // TODO - for data exporting, check if we must export also something else
      .zipWithIndex
      .foreach { case (reward, index) =>
        applicationNodes(index).setConcentration(new SimpleMolecule(Molecules.reward), reward.asInstanceOf[T])
      }
    torch.Tensor(rewards.toPythonProxy)
  }

  private def getComponents: Seq[Component] =
    getAllocator(applicationNodes.head)
      .getComponentsAllocation
      .keys
      .map(id => Component(id))
      .toSeq

  private def getAllocator(node: Node[T]) = {
    node.getProperties.asScala
      .filter(_.isInstanceOf[AllocatorProperty[T, P]])
      .map(_.asInstanceOf[AllocatorProperty[T, P]])
      .head
  }
}