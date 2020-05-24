library(shiny)
library(shinyWidgets)

#Front-End
shinyUI(fluidPage(
  titlePanel(title = "Animations"),
  sidebarLayout(
    sidebarPanel(h3("Select the Features"),
                 selectInput("map", "Select a map from the drop down list", maplist),
                 pickerInput("users", "Select user(s)", choices = userlist, selected = userlist, options = list('actions-box' = TRUE), multiple = TRUE),
                 p("Please do not deselect all users and click on render as it crashes the app"),
                 sliderInput("nframes", "Select the number of frames", min = 10, max = 50, value = 25),
                 p("Please keep in mind that higher number of frames will result in longer rendering times."),
                 sliderInput("fps", "Select frames per second", min = 1, max = 50, value = 10),
                 radioButtons("color", "Select the coloring of the datapoints", choices = c("user", "user.index"), selected = "user"),
                 fileInput("backImage", "Select a Map for Background", accept = c("image/jpeg", "image/png"), multiple = FALSE),
                 actionButton("Load", "Render")
                 ),
    
    mainPanel(h4("Please Wait While Animations Are Being Rendered"),
      #imageOutput("mymap"),
      imageOutput("plot1")
      )
  )
))


