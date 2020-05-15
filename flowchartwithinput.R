library(shiny)
library(cluster)    # clustering algorithms
library(factoextra) # clustering algorithms & visualization
library(dplyr)
library(markovchain)
library(lattice)
library(diagram)

tracking_data <- read.csv('metro_data.csv', sep = ';')
stim_list = c(tracking_data$StimuliName)
stim_list_unique <- unique(stim_list)


shinyApp(
  ui = fluidPage(selectInput("variable", "Variable:", stim_list_unique),
      mainPanel(
        plotOutput("plot")
      )
#    )
  ),
  server = function(input, output) {
    output$plot <- renderPlot({
      input$newplot
      subs <- subset(tracking_data, StimuliName == input$variable)
      cluster_df_inp <- select(subs, "MappedFixationPointX","MappedFixationPointY")
      
      k2_inp <- kmeans(cluster_df_inp, centers = 6, nstart  = 25)
      
      subs[["Cluster"]] <- c(k2_inp$cluster)
      
      user_list_inp = c(subs$user)
      user_list_unique_inp <- unique(user_list_inp)
      
      startmatrix_inp <- matrix(nrow = 6, ncol = 6)
      startmatrix_inp[] <- 0L
      for (n in user_list_unique_inp) {
        for_seq <- c(subset(subs, user == n)$Cluster)
        sequenceMatr_for <-createSequenceMatrix(for_seq,sanitize=FALSE, possibleStates = seq(1:6))
        startmatrix_inp <- startmatrix_inp + sequenceMatr_for
      }
      plot <- plotmat(A = as.data.frame(startmatrix_inp), pos = 6, curve = 0.7, name = name, lwd = 2, arr.len = 0.6, arr.width = 0.25, my = -0.2, box.size = 0.05, arr.type = "triangle", dtext = 0.95, main = "Flow chart AOI")
    })
  }
)


