package learning.model

/** A mathematical function that decreases its value over the time */
trait Decay[T]{
  def update(): Unit
  def value(): T
}

/** A decay that decreases its value exponentially
 *
 * @param initialValue the initial value of the sequence
 * @param rate the rate at which values decrease
 * @param bound the lower bound of the sequence
 */
class ExponentialDecay(initialValue: Double, rate: Double, bound: Double) extends Decay[Double]{

  private var elapsedTime: Int = 0
  override def update(): Unit = elapsedTime = elapsedTime + 1
  override def value(): Double = {
    val v = initialValue * Math.pow(1 - rate, elapsedTime)
    if (v > bound) { v } else { bound }
  }

}