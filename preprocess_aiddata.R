library(readr)
library(data.table)

# setwd("C:/git/iati_ag")
setwd("/media/alex/Windows/git/iati_ag")
source("preprocess_data.R")

datasets <- c(
  "aiddata/project__1_2017-09-22-13-53.csv"
  ,"aiddata/project__2_2017-09-22-16-03.csv"
  ,"aiddata/project__3_2017-09-22-18-15.csv"
  ,"aiddata/project__4_2017-09-22-19-47.csv"
)

datList <- list()
datIndex <- 1

for(dataset in datasets){
  dat <- read_csv(dataset)
  datList[[datIndex]] <- dat
  datIndex <- datIndex + 1
}

dat <- rbindlist(datList)
dat$description <- paste(dat$Title,dat$`Short Description`,dat$`Long Description`)
dat$description <- gsub("/n"," ",dat$description)
dat$description <- gsub("\\s+"," ",dat$description)
dat <- subset(dat,!is.na(description))
setnames(dat,"AidData Purpose Code","sector")
keep <- c("description","sector")
dat <- data.frame(dat)[keep]

dat <- subset(dat,grepl("311[0-9][0-9]",sector))

sectors <- unique(dat$sector)

iati <- read_csv("iati_ag_preprocess.csv")
iati <- subset(iati,sector %in% sectors)
iati_sectors <- unique(iati$sector)
dat <- subset(dat,sector %in% iati_sectors)
dat <- rbind(dat,iati)

dat <- unique(dat)

write.csv(dat,"iati_ag_preprocess.csv",na="",row.names=FALSE)
