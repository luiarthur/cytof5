import ammonite.ops._

def dirContainsJld(d: Path):Boolean = {
  (ls.rec! d).map(_.last.ext == "jld2").contains(true)
}

val dirsWithJld = (ls! 'results).filter( _.isDir ).filter( dirContainsJld )
dirsWithJld.foreach{ println }

dirsWithJld.foreach{ d => 
  try {
    %('julia, "post_process.jl", d)
  } catch {
    case _ : Throwable => println("Not successful for ${d.toString}")
  }
}

