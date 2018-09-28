object MakeArgs {
  val jlArgs = """
    I
    J
    nFac
    K
    L
    K_MCMC
    L_MCMC
    b0Sd
    b1Scale
    SEED
    """.split("\n").map{ _.trim }.filter{_.size > 0}



  def create(between:String, within:String, before:String=""):String = {
    jlArgs.map{a => s"${before}${a}${within}$${${a}}"}.mkString(between)
  }

  def main(args:Array[String]) {
    val out = args.head match {
      case "cmd" => create(between=" ", within="=", before="--")
      case "exp" => create(between="_", within="")
      case _ => "Not supported. Try 'cmd' or 'exp'"
    }

    println(out)
  }
}


