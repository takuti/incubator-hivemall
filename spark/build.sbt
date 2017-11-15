scalaVersion := "2.11.8"

licenses += "Apache-2.0" -> url("http://opensource.org/licenses/Apache-2.0")

javaOptions ++= Seq("-Xms256m", "-Xmx1024m", "-XX:MaxPermSize=1024m", "-XX:+CMSClassUnloadingEnabled")

net.virtualvoid.sbt.graph.Plugin.graphSettings

parallelExecution in Test := false

// To avoid compiler errors in sbt-doc
// sources in doc in Compile := List()

// To skip unit tests in assembly
test in assembly := {}

// spark-package settings
spName := "maropu/hivemall-spark"

sparkVersion := "1.6.1"

sparkComponents ++= Seq("sql", "mllib", "hive")

// Copied to handle compatibility stuffs for Hive UDFs
unmanagedSourceDirectories in Compile += baseDirectory.value / "extra-src/hive"

// credentials += Credentials(Path.userHome / ".ivy2" / ".sbtcredentials")

// resolvers += Resolver.sonatypeRepo("releases")
// addCompilerPlugin("org.scalamacros" % "paradise" % "2.1.0-M5" cross CrossVersion.full)

libraryDependencies ++= Seq(
  "org.apache.commons" % "commons-compress" % "1.8",
  "io.github.myui" % "hivemall-core" % "0.4.2-rc.4",
  "io.github.myui" % "hivemall-mixserv" % "0.4.2-rc.4",
  "org.scalatest" % "scalatest_2.11" % "2.2.4" % "provided",
  "org.xerial" % "xerial-core" % "3.2.3" % "provided"
)

mergeStrategy in assembly := {
  case PathList("javax", "servlet", xs @ _*) =>
    MergeStrategy.first
  case PathList(ps @ _*) if ps.last endsWith ".properties" =>
    MergeStrategy.first
  case PathList(ps @ _*) if ps.last endsWith ".html" =>
    MergeStrategy.first
  case PathList(ps @ _*) if ps.last endsWith ".xml" =>
    MergeStrategy.first
  case PathList(ps @ _*) if ps.last endsWith ".types" =>
    MergeStrategy.first
  case PathList(ps @ _*) if ps.last endsWith ".class" =>
    MergeStrategy.first
  case PathList(ps @ _*) if ps.last endsWith ".thrift" =>
    MergeStrategy.first
  case "application.conf" =>
    MergeStrategy.concat
  case "unwanted.txt" =>
    MergeStrategy.discard
  case x =>
    val oldStrategy = (mergeStrategy in assembly).value
    oldStrategy(x)
}

