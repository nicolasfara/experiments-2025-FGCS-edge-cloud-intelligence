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
) extends GraphBuilderReaction[T, P](environment, distribution) {

  // TODO - check that nodes have the right molecule inside (application/infrastructural)

  // TODO - to parameters
  private val edgeServerSize = 5
  private val components = Seq(Component("it.unibo.Gradient"), Component("it.unibo.LeaderElection")) // TODO - add real components


  private var oldGraph: Option[py.Dynamic] = None
  private var oldActions: Option[py.Dynamic] = None
  private val rewardFunction = rlUtils.BatteryRewardFunction()
  private val actionSpace = ActionSpace(components, edgeServerSize)

  private lazy val learner: py.Dynamic = environment
    .getLayer(new SimpleMolecule(Molecules.learner))
    .asInstanceOf[LearningLayer[P]]
    .getValue(environment.makePosition(0, 0))

  override protected def getNodeFeature(node: Node[T]): Vector = {
    val position = environment.getPosition(node)
    Vector(Seq(position.getCoordinate(0), position.getCoordinate(0)))
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
        node.getProperties.asScala
          .filter(_.isInstanceOf[AllocatorProperty[T, P]])
          .map(_.asInstanceOf[AllocatorProperty[T, P]])
          .head
          .setComponentsAllocation(newComponentsAllocation)
      }

    (oldGraph, oldActions) match {
      case (Some(previousObs), Some(previousActions)) =>
        val rewards = computeRewards(previousObs, observation)
        learner.add_experience(previousObs, previousActions, rewards, observation)
        learner.train_step_dqn(batch_size=32, gamma=0.99, update_target_every=10)
      case _ =>
    }

    oldGraph = Some(observation)
    oldActions = Some(actions)
  }

  private def computeRewards(obs: py.Dynamic, nextObs: py.Dynamic): py.Dynamic = {
    val rewards = rewardFunction.compute(obs, nextObs).tolist().as[List[Int]]
    rewards // TODO - for data exporting, check if we must export also something else
      .zipWithIndex
      .foreach { case (reward, index) =>
        applicationNodes(index).setConcentration(new SimpleMolecule(Molecules.reward), reward.asInstanceOf[T])
      }
    torch.Tensor(rewards.toPythonProxy)
  }

}

/**
 * For each component we have a set of possible actions C0_actions = {C0_A, C0_ES1, ..., C0_ESn}
 * The complete action space is the cartesian product {(C0_A,C1_A), (C0_A, C1_ES1), ...}
 * The cardinality of the action space is |C0_actions| * ... * |Ck_actions|
 * */
