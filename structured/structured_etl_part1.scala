import org.apache.spark.{SparkConf, SparkContext}

val conf = new SparkConf()
  .setAppName("SparkMe Application")
  .setMaster("local[*]")

val sc = new SparkContext(conf)

case class ChartEvents(rowID: String, SubjectID: String, hadmID: String, icustayID: String, itemID:String, charttime:String, storetime:String, cgID:String, value: String, valuenum: String, valueuom: String)

val chartevents = sc.textFile("file:///mnt/host/c/Users/yeswe/Documents/SP18/CSE6250/project/data/CHARTEVENTS.csv")
    .map(line => line.split(","))
    .filter(line => !line(0).contains("ROW_ID"))
    .filter(line=> (line.length > 10))
    .map(line => ChartEvents(line(0), line(1), line(2), line(3), line(4), line(5), line(6), line(7), line(8), line(9), line(10)))
val sce = chartevents

def parseDouble(s: String) = {
    try { Some(s.toDouble)    true}
    catch { case _ => None
        false}
    }

val iddict = sce.map(x => (x.itemID, x.value, x.valuenum)).filter(x => !x._3.isEmpty).map(x=> (x._1)).distinct.zipWithIndex
val im = iddict.collectAsMap
var maxIndex = iddict.map(_._2).max()
val iddict2 = sce.map(x => (x.itemID, x.value, x.valuenum)).filter(x => x._3.isEmpty).map(x=> (x._1, x._2)).distinct.zipWithIndex.map(x=> (x._1, x._2+maxIndex))
val i2m = iddict2.collectAsMap
val map = im ++ i2m.map{ case (k,v) => k -> (v) }

def replacedigs(n:String) = {map.getOrElse(n,"")}
def replacefacs(n:(String,String)) = {(map.getOrElse(n,""),1.0)}
val digs = sce.filter(x => !x.valuenum.isEmpty).filter(x => parseDouble(x.valuenum)).map(x => (x.hadmID, (replacedigs(x.itemID), x.valuenum.toDouble)))
val facs = sce.filter(x => x.valuenum.isEmpty).map(x => (x.hadmID, replacefacs(x.itemID, x.value)))

val u = digs.union(facs)

val gbk = u.groupByKey()
val finalish = gbk.map(x => (x._1,x._2.groupBy(_._1).map{case (k,v)=>(k, v.map(_._2).sum)}.toList))
//finalish.saveAsTextFile("file:///mnt/host/c/Users/yeswe/Documents/SP18/CSE6250/project/data/sparserep")
val admissions = sc.textFile("file:///mnt/host/c/Users/yeswe/Documents/SP18/CSE6250/project/data/ADMISSIONS.csv")
	.map(line => line.split(","))
	.filter(line => !line(0).contains("ROW_ID"))

//admissions.take(3).map(x => x.toList).foreach(println)

var m = map

val admtype = admissions.map(x => x(6)).distinct.zipWithIndex.map(x => (x._1, x._2+maxIndex)).collectAsMap
m = m ++ admtype
maxIndex = m.keySet.size

val admloc = admissions.map(x => x(7)).distinct.zipWithIndex.map(x => (x._1, x._2+maxIndex)).collectAsMap
m = m ++ admloc
maxIndex = m.keySet.size

val admdisloc = admissions.map(x => x(8)).distinct.zipWithIndex.map(x => (x._1, x._2+maxIndex)).collectAsMap
m = m ++ admdisloc
maxIndex = m.keySet.size

val admins = admissions.map(x => x(9)).distinct.zipWithIndex.map(x => (x._1, x._2+maxIndex)).collectAsMap
m = m ++ adminsmaxIndex = m.keySet.size

val admlang = admissions.map(x => x(10)).distinct.zipWithIndex.map(x => (x._1, x._2+maxIndex)).collectAsMap
m = m ++ admlang
maxIndex = m.keySet.size

val admrel = admissions.map(x => x(11)).distinct.zipWithIndex.map(x => (x._1, x._2+maxIndex)).collectAsMap
m = m ++ admrel
maxIndex = m.keySet.size

val admmar = admissions.map(x => x(12)).distinct.zipWithIndex.map(x => (x._1, x._2+maxIndex)).collectAsMap
m = m ++ admmar
maxIndex = m.keySet.size

val admeth = admissions.map(x => x(13)).distinct.zipWithIndex.map(x => (x._1, x._2+maxIndex)).collectAsMap
m = m ++ admeth
maxIndex = m.keySet.size

val admdiag = admissions.map(x => x(16)).distinct.zipWithIndex.map(x => (x._1, x._2+maxIndex)).collectAsMap
m = m ++ admdiag
maxIndex = m.keySet.size

//admins.take(15).foreach(println)
//admissions.map(x => (x(2),(x(6),x(7)))).join(finalish).take(15).foreach(println)

val enddata = admissions.map(x => (x(2),(x(6),x(7),x(8),x(9),x(10),x(11),x(12),x(13),x(16)))).join(finalish).map(x => (x._1, x._2._2.toList :+ (m.getOrElse(x._2._1._1,""),1.0) :+ (m.getOrElse(x._2._1._2,""),1.0):+ (m.getOrElse(x._2._1._3,""),1.0):+ (m.getOrElse(x._2._1._4,""),1.0):+ (m.getOrElse(x._2._1._5,""),1.0):+ (m.getOrElse(x._2._1._6,""),1.0):+ (m.getOrElse(x._2._1._7,""),1.0):+ (m.getOrElse(x._2._1._8,""),1.0) :+ (m.getOrElse(x._2._1._9,""),1.0)))
enddata.map(x => (x._1,x._2.groupBy(_._1).map{case (k,v)=>(k, v.map(_._2).sum)}.toList)).saveAsTextFile("file:///mnt/host/c/Users/yeswe/Documents/SP18/CSE6250/project/data/sparserep")