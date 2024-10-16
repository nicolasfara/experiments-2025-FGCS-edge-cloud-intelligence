package it.unibo.alchemist.model

import me.shadaj.scalapy.py

class LearningLayer[P <: Position[P]](learner: py.Dynamic) extends Layer[py.Any, P] {
  override def getValue(p: P): py.Dynamic = learner
}
