package learning.model

/**
 * For each component we have a set of possible actions C0_actions = {C0_A, C0_ES1, ..., C0_ESn}
 * The complete action space is the cartesian product {(C0_A,C1_A), (C0_A, C1_ES1), ...}
 * The cardinality of the action space is |C0_actions| * ... * |Ck_actions|
 * */

case class Component(id: String)
sealed trait Device
case object MySelf extends Device
case object EdgeServer extends Device
case object Cloud extends Device


case class Cloud() extends Device

case class PairComponentDevice(component: Component, device: Device)

case class ActionSpace(components: Seq[Component]){

  private val locations = Seq(MySelf, EdgeServer, Cloud)

  private val pairs = components
    .map(c =>
      locations.map(l => PairComponentDevice(c, l))
    )

  val actions: Seq[List[PairComponentDevice]] =
    pairs.foldLeft(Seq(List.empty[PairComponentDevice])) {
      (acc, seq) => for {
        combo <- acc
        element <- seq
      } yield combo :+ element
    }
}
object Main extends App {
  val as = ActionSpace(Seq(Component("Gradient"), Component("Greater")))
  as.actions.foreach(println(_))
  println(as.actions.size)
}