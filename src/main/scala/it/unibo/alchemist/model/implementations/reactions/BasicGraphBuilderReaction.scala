package it.unibo.alchemist.model.implementations.reactions

import it.unibo.alchemist.model.{Environment, Node, Position, TimeDistribution}

import scala.jdk.CollectionConverters.CollectionHasAsScala
import it.unibo.alchemist.model.molecules.SimpleMolecule
import it.unibo.alchemist.utils.PythonModules.{rlUtils, torch}
import it.unibo.alchemist.utils.Molecules

import scala.language.implicitConversions
import me.shadaj.scalapy.py.SeqConverters
import me.shadaj.scalapy.py
import me.shadaj.scalapy.readwrite.{Reader, Writer}

class BasicGraphBuilderReaction[T, P <: Position[P]](
    environment: Environment[T, P],
    distribution: TimeDistribution[T],
) extends GraphBuilderReaction[T, P](environment, distribution) {

  override protected def getNodeFeature(node: Node[T]): Vector = {
    val position = environment.getPosition(node)
    Vector(Seq(position.getCoordinate(0), position.getCoordinate(0)))
  }

  override protected def getEdgeFeature(node: Node[T], neigh: Node[T]): Vector = {
    val distance = environment.getPosition(node).distanceTo(environment.getPosition(neigh))
    Vector(Seq(distance))
  }
}
