library(readr)
library(data.table)

# setwd("C:/git/iati_ag")
setwd("/media/alex/Windows/git/iati_ag")

dat <- read_csv("iati_agriculture.csv")

dat <- subset(dat,!is.na(description))

multiSubStr <- function(x,str,sslength=4){
  return(substr(str,x,x+sslength))
}

sectorSplitList <- list()
sectorSplitIndex <- 1

for(i in 1:nrow(dat)){
  sectorCode <- dat[i,"sector-code"][[1]]
  description <- dat[i,"description"][[1]]
  sectorSplit <- sapply(gregexpr("311[0-9][0-9]",sectorCode)[[1]],multiSubStr,str=sectorCode)
  sectorDat <- data.frame(description,sector=sectorSplit)
  sectorSplitList[[sectorSplitIndex]] <- sectorDat
  sectorSplitIndex <- sectorSplitIndex + 1
}

sectorDf <- rbindlist(sectorSplitList)
sectorDf <- unique(sectorDf)
table(sectorDf$sector)

write.csv(sectorDf,"iati_ag_preprocess.csv",na="",row.names=FALSE)
