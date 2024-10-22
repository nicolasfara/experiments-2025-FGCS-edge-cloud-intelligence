package it.unibo.alchemist.model

import learning.model.ExponentialDecay

class DecayLayer [P <: Position[P]](decay: ExponentialDecay) extends Layer[ExponentialDecay, P]{

  override def getValue(p: P): ExponentialDecay = decay

}
