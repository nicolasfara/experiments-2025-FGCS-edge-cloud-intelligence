package it.unibo.alchemist.boundary.extractors
import it.unibo.alchemist.model
import it.unibo.alchemist.model.molecules.SimpleMolecule
import it.unibo.alchemist.model.{Actionable, Environment, Position}

import java.{lang, util}
import scala.jdk.CollectionConverters._
import scala.language.implicitConversions

class MoleculePerNodeExtractor[T, P <: Position[P]](
    private val environment: Environment[T, P],
    private val moleculeName: String,
    private val precision: Int,
) extends AbstractDoubleExporter(precision) {
  private lazy val molecule = new SimpleMolecule(moleculeName)

  override def getColumnNames: util.List[String] =
    environment.getNodes
      .iterator()
      .asScala
      .toList
      .map(n => s"node-${n.getId}[$moleculeName]")
      .asJava

  override def extractData[T](
      environment: Environment[T, _],
      actionable: Actionable[T],
      time: model.Time,
      l: Long,
  ): util.Map[String, lang.Double] =
    environment.getNodes
      .iterator()
      .asScala
      .map(n => s"node-${n.getId}[$moleculeName]" -> Option(n.getConcentration(molecule).asInstanceOf[Double]).getOrElse(Double.NaN))
      .toMap
      .view
      .mapValues(Double.box)
      .toMap
      .asJava
}
