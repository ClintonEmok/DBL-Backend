library(shiny)
library(ggplot2)
library(gganimate)
library(shinyWidgets)
library(stringr)

#Back-End
shinyServer(
  function(input, output, session){
    #Importing & Preprocessing Data
    metro_data <- read.csv('metro_data.csv', sep = ';', encoding = "latin-1")
    maplist <- unique(metro_data$StimuliName)
    userlist <- metro_data$user %>% unique() %>% as.character()
    userlist <- str_sort(userlist, numeric = TRUE)
    
#    inFile <- reactive({
#      inFile <- input$backImage
#      inFile$datapath <- gsub("\\\\", "/", inFile$datapath)
#      inFile
#    })
#    observeEvent(input$backImage, {
#      inFile <- input$backImage
#      if(is.null(inFile)){
#        return()
#      }  
#      file.copy(inFile$datapath, file.path("C:/Users/20190771/Documents/GitHub/DBL-Backend/R/DBL/Server Upload", inFile$name))
#    })
    
    data1 <- reactive({
      #creating the index column
      mapdata <- subset(metro_data, StimuliName == input$map)
      input$Load
      isolate({mapdata <- mapdata[mapdata$user %in% input$users,]})
      mapdata$index <- 1:nrow(mapdata)
        
      #creating the user index column
      mapdata$user.index <- 1:nrow(mapdata)
      
      user.count <- 1
      for(i in 2:nrow(mapdata)){
        if(mapdata[i, 7] == mapdata[i-1, 7]){
          user.count <- user.count + 1
          mapdata[i, 10] <- user.count
        } else {
          user.count <- 1
          mapdata[i, 10] = user.count
        }
      }
      
      return(mapdata)
    })

    
    observe({
      maxFrame <- data1()[data1()$user.index == max(data1()$user.index), 10]
      updateSliderInput(session, "nframes", max = maxFrame)
      
      userlist = data1()$user %>% unique() %>% as.character()
      userlist = str_sort(userlist, numeric = TRUE)
      
      updatePickerInput(session, "users", choices = userlist, selected = userlist)
    })
    
    output$mymap <- renderText({
      paste("The current nframes is: ", input$nframes)
    })
  

    # Creting Plot
    output$plot1 <- renderImage({
      req(input$Load)
      input$Load
      isolate({
        outfile <- tempfile(fileext = '.gif')
        
        mapplot <- ggplot(data1(), aes_string(x = "MappedFixationPointX", y = "MappedFixationPointY", color = input$color)) +
          #Graph
          #background_image(inFile) +
          geom_point(aes(size = FixationDuration), alpha = 0.8) + 
          coord_fixed() +
          scale_size(range = c(1,16)) +
          scale_y_reverse() +
          transition_time(user.index) +
          shadow_mark(PAST = TRUE) +
          labs(title = "Frame {frame} of {nframes}") + 
          theme(plot.title = element_text(size = 18, face ="bold"))
        
        anim_save("outfile.gif", animate(mapplot, nframes = (input$nframes + input$fps*2), fps = input$fps, end_pause = input$fps*2, width = 800, height = 600))
        
        list(src = "outfile.gif",
             contentType = 'image/gif')
      })
    })

  }
)


