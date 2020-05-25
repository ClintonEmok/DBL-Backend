library(shiny)
library(cluster)    # clustering algorithms
library(factoextra) # clustering algorithms & visualization
library(markovchain)
library(ggplot2)
library(dplyr)
library(diagram)

tracking_data <- read.csv('metro_data.csv', sep = ';')
stim_list_unique <- unique(tracking_data$StimuliName)

shinyApp(
  ui = fluidPage(
    selectInput("map", "Map:", stim_list_unique),
    numericInput('clusters', 'Cluster count', 3, min = 1, max = 9),
    mainPanel(
      fluidRow(
        splitLayout(cellWidths = c("50%", "50%"), plotOutput("plot1"), plotOutput("plot2"))
      )
    )
  ),
  
  
  server = function(input, output) {
    selectedData <- reactive({
      dplyr::select(subset(tracking_data, StimuliName == input$map), "MappedFixationPointX","MappedFixationPointY")
    })
    
    clusters <- reactive({
      set.seed(123)
      kmeans(selectedData(), centers = input$clusters, nstart  = 25)
    })
    
    output$plot1 <- renderPlot({
      fviz_cluster(clusters(), data=selectedData())
    })
    
    output$plot2 <- renderPlot({
      input$newplot
      set.seed(123)
      subs <- subset(tracking_data, StimuliName == input$map)
      cluster_df_inp <- dplyr::select(subs, "MappedFixationPointX","MappedFixationPointY")
      
      k2_inp <- kmeans(cluster_df_inp, centers = input$clusters, nstart = 25)
      
      subs[["Cluster"]] <- c(k2_inp$cluster)
      
      user_list_inp = c(subs$user)
      user_list_unique_inp <- unique(user_list_inp)
      
      startmatrix_inp <- matrix(nrow = input$clusters, ncol = input$clusters)
      startmatrix_inp[] <- 0L
      n <- c(1:6)
      for (n in user_list_unique_inp) {
        for_seq <- c(subset(subs, user == n)$Cluster)
        sequenceMatr_for <-createSequenceMatrix(for_seq,sanitize=FALSE, possibleStates = n)
        startmatrix_inp <- startmatrix_inp %+% sequenceMatr_for
      }
      plot2 <- diagram::plotmat(A = as.data.frame(startmatrix_inp), pos = 6, curve = 0.7, name = name, lwd = 2, arr.len = 0.6, arr.width = 0.25, my = -0.2, box.size = 0.05, arr.type = "triangle", dtext = 0.95, main = "Flow chart AOI") 
    })
    
  }
)

