set.seed(123)

library(tidyverse)  # data manipulation
library(cluster)    # clustering algorithms
library(factoextra) # clustering algorithms & visualization
library(dplyr)
tracking_data <- read.csv('metro_data.csv', sep = ';')

library(markovchain)
library(lattice)
library(diagram)
library(networkD3)

#eyetr_data2 <- read.csv("metro_data.csv", TRUE, ",")
ant_s1 <- subset(tracking_data, StimuliName == "01_Antwerpen_S1.jpg")


cluster_df <- select(ant_s1, "MappedFixationPointX","MappedFixationPointY")
head(cluster_df)

k2 <- kmeans(cluster_df, centers = 6, nstart  = 25)

cl_ant_s1 <- ant_s1
cl_ant_s1[["Cluster"]] <- c(k2$cluster)
head(cl_ant_s1)

user_list = c(ant_s1$user)
user_list_unique <- unique(user_list)

startmatrix <- matrix(nrow = 6, ncol = 6)
startmatrix[] <- 0L

for (n in user_list_unique) {
  print(n)
  for_seq <- c(subset(cl_ant_s1, user == n)$Cluster)
  sequenceMatr_for <-createSequenceMatrix(for_seq,sanitize=FALSE, possibleStates = seq(1:6))
  startmatrix <- startmatrix + sequenceMatr_for
  print(startmatrix)
}
print(startmatrix)

      
