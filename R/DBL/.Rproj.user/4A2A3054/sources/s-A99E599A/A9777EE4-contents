library(shiny)
library(ggplot2)
library(gganimate)
library(shinyWidgets)
library(stringr)
library(jpeg)
library(ggpubr)
library(rsconnect)
library(gifski)

#Back-End
shinyServer(
  function(input, output, session){
    #Importing & Preprocessing Data
    metro_data <- read.csv('metro_data.csv', sep = ';', encoding = "latin-1")
    maplist <- unique(metro_data$StimuliName)
    userlist <- metro_data$user %>% unique() %>% as.character()
    userlist <- str_sort(userlist, numeric = TRUE)
    

#    inFile <- observeEvent(input$backImage, {
#      if(is.null(input$backImage)){
#        return()
#      } else {
#        inp <- input$backImage
#        inFile <- jpeg::readJPEG(inp$datapath)
#        return(inFile)
#      }
#    }) 
    
    
    data1 <- reactive({
      mapdata <- subset(metro_data, StimuliName == input$map)
      
      userlist =  mapdata$user %>% unique() %>% as.character()
      userlist = str_sort(userlist, numeric = TRUE)
      updatePickerInput(session, "users", choices = userlist, selected = userlist)
      
      input$Load
      isolate({mapdata <- mapdata[mapdata$user %in% input$users,]})
      mapdata$index <- 1:nrow(mapdata)
        
      #creating the index column
      mapdata$user.index <- 1:nrow(mapdata)
      
      #creating user index column
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
      
    })
    
  

    # Creting Plot
    output$plot1 <- renderImage({
      req(input$Load)
      input$Load
      isolate({
        outfile <- tempfile(fileext = '.gif')
        
        mapplot <- ggplot(data1(), aes_string(x = "MappedFixationPointX", y = "MappedFixationPointY", color = input$color)) +
          #Graph
          geom_point(aes(size = FixationDuration), alpha = 0.8) + 
          coord_fixed() +
          scale_size(range = c(1,16)) +
          scale_y_reverse() +
          transition_time(user.index) +
          shadow_mark(PAST = TRUE) +
          labs(title = "Frame {frame} of {nframes}") + 
          theme(plot.title = element_text(size = 18, face ="bold"))
        
        #if(!is.null(inFile)){
        #  mapplot <- mapplot + background_image(inFile)
        #}
        
        anim_save("outfile.gif", animate(mapplot, nframes = (input$nframes + input$fps*2), fps = input$fps, end_pause = input$fps*2, width = 800, height = 600,
                                         renderer = gifski_renderer()))
        
        list(src = "outfile.gif",
             contentType = 'image/gif')
      })
    })

  }
)


