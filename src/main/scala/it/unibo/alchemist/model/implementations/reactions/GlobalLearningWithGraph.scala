package it.unibo.alchemist.model.implementations.reactions

import it.unibo.alchemist.model.molecules.SimpleMolecule
import it.unibo.alchemist.model._
import me.shadaj.scalapy.py

import scala.jdk.CollectionConverters.CollectionHasAsScala
import scala.language.implicitConversions

class GlobalLearningWithGraph[T, P <: Position[P]](
    environment: Environment[T, P],
    distribution: TimeDistribution[T],
) extends GraphBuilderReaction[T, P](environment, distribution) {
  var oldGraph: Option[py.Any] = None
  lazy val learner = environment.getLayer(new SimpleMolecule("learner")).asInstanceOf[LearningLayer[P]].getValue(environment.makePosition(0, 0))
  override protected def getNodeFeature(node: Node[T]): Vector = {
    val position = environment.getPosition(node)
    Vector(Seq(position.getCoordinate(0), position.getCoordinate(0)))
  }

  override protected def getEdgeFeature(node: Node[T], neigh: Node[T]): Vector = {
    val distance = environment.getPosition(node).distanceTo(environment.getPosition(neigh))
    Vector(Seq(distance))
  }

  override protected def handleGraph(graph: py.Any): Unit = {
    val rewards = environment.getNodes.asScala.map { node =>
      val reward = node.getConcentration(new SimpleMolecule("reward")) // or something else??
    }
    // rewards to tensors
    val tensorRewards = rewards // todo convert to tensor
    val actions = learner.select_action(graph, 0.05) // TODO 0.1 need to be injected from the outside
    // globally perform action
    // => put some data on molecules
    /**
     * Alternative 1:
     * components {C_0, C_1}
     * action space {C_O_offloading, C_0_onloading, C_1_offloading, C_1_onloading}
     * |action space| = |components| * 2
     * nodes [0, 1, 2, 3, 4]
     * actions [1, 0, 2, 1]
     * actions_semantics [C_0_onloading, C_0_offloading, C_0_offloading, C_0_onloading]
     * Alternative 2:
     * components {C_0, C_1}
     * Edge Servers {A, ES_0, ES_1}
     * action space { C_0_offloading_A, C_0_offloading_ES_0, C_0_offloading_ES_1, C_1_offloading_A, C_1_offloading_ES_0, C_1_offloading_ES_1}
     * |action space| = (|Edge Servers| + 1) ^ |components|
     */
    oldGraph match {
      case Some(old) =>
        // learner.add_experience(old, actions, tensorRewards, graph)
        learner.train_step_dqn()
    }
  }

}
