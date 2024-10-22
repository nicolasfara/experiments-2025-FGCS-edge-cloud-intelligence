package it.unibo.alchemist.model

import learning.model.ExponentialDecay

class DecayLayer [P <: Position[P]](decay: Double) extends Layer[Double, P]{

  override def getValue(p: P): Double = decay

}
