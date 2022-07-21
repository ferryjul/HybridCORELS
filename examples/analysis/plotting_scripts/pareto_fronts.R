library(ggplot2)
library(scales)

point_size <- 5.5
line_size <- 0.5
point_size_vline <- 1.5

alpha_line <- 0.99
alpha_point <- 1.0
alpha_ribbon <- 0.15


fig_width       <- 48
fig_height      <- 24

basesize <- 70 


pareto_plot <- function(dataset, method) {
    input_file <- sprintf("../results/pareto/%s/%s.csv", method, dataset)
    save_path <- "./graphs/pareto"
    dir.create(save_path, showWarnings = FALSE, recursive = TRUE)
    output_file <- sprintf("%s/%s_%s.jpg", save_path, dataset, method)
    df  <- read.csv(input_file, header=T)

    pp <- ggplot() + 
    geom_line(data=df, aes(x=transparency, y=accuracy),  size=line_size, linetype = "dashed", alpha=alpha_line) + 
    geom_point(data=df, aes(x=transparency, y=accuracy), size=point_size, alpha=alpha_point) + 
    theme_minimal(base_size=basesize) + 
    labs(x = "Transparency", y = "Accuracy") 

    pp 
    ggsave(output_file, dpi=300, width=fig_width, height=fig_height)              
}






datasets <- c("compas")
methods  <- c("crl")
for (dataset in datasets){
    for (method in methods){
            pareto_plot(dataset, method)
    }
}
