package learning.model

/**
 * For each component we have a set of possible actions C0_actions = {C0_A, C0_ES1, ..., C0_ESn}
 * The complete action space is the cartesian product {(C0_A,C1_A), (C0_A, C1_ES1), ...}
 * The cardinality of the action space is |C0_actions| * ... * |Ck_actions|
 * */

case class Component(id: String)
sealed trait Device
case class MySelf() extends Device
case class EdgeServer(id: Int) extends Device

case class PairComponentDevice(component: Component, device: Device)

case class ActionSpace(components: Seq[Component], devicesCardinality: Int){

  private val devices = Range(0, devicesCardinality)
    .map(i => EdgeServer(i))
    .prepended(MySelf())

  private val pairs = components.map(c => devices.map(d => PairComponentDevice(c, d)))

  val actions: Seq[List[PairComponentDevice]] =
    pairs.foldLeft(Seq(List.empty[PairComponentDevice])) {
      (acc, seq) => for {
        combo <- acc
        element <- seq
      } yield combo :+ element
    }
}
object Main extends App {
  ActionSpace(Seq(Component("Gradient"), Component("Greater")), 3).actions.foreach(println(_))
  println(ActionSpace(Seq(Component("Gradient"), Component("Greater")), 3).actions.size)
}