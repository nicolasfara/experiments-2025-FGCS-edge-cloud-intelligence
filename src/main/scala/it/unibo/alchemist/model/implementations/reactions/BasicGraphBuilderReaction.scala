package it.unibo.alchemist.model.implementations.reactions

import it.unibo.alchemist.model.{Environment, LearningLayer, Node, Position, TimeDistribution}

import scala.jdk.CollectionConverters.CollectionHasAsScala
import it.unibo.alchemist.model.molecules.SimpleMolecule
import it.unibo.alchemist.utils.PythonModules.{rlUtils, torch}
import it.unibo.alchemist.utils.Molecules

import scala.language.implicitConversions
import me.shadaj.scalapy.py.SeqConverters
import me.shadaj.scalapy.py
import me.shadaj.scalapy.readwrite.{Reader, Writer}

class GlobalLearningWithGraph[T, P <: Position[P]](
    environment: Environment[T, P],
    distribution: TimeDistribution[T],
) extends GraphBuilderReaction[T, P](environment, distribution) {
  var oldGraph: Option[py.Any] = None
  lazy val learner = environment.getLayer(new SimpleMolecule("learner")).asInstanceOf[LearningLayer[P]].getValue(environment.makePosition(0,0))
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
    val tensorRewards = rewards // todo
    val actions = learner.select_action(graph, 0.05) // 0.1 need to be injected from the outside
    oldGraph match {
      case Some(old) =>
        learner.add_experience(old, actions, tensorRewards, graph)
        learner.train_step_dqn()
    }
  }

}
