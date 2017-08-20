#!/usr/bin/env Rscript

## Copyright: 2017
## Author: C. Clayton Violand

## Collects the following results from file and compiles into a nice table:
##
## I. Accuracies
## II. Speed
## III. Memory Utilization
##

options(warn = -1)

library(gtools)

modes <- c("caffe", "eigen", "gemmlowp")
models <- list.files("models", pattern = "[^[README.md]]*")

################
## Accuracies ##
################
# prepare table of results.
data <- do.call(rbind, lapply(models, function(model) {  # for each model...
    data.set <- do.call(rbind, lapply(modes, function(mode) {   # and for each mode...
        scores.path <- file.path("features", model, mode)
        files <- list.files(scores.path, full.names = TRUE)
        files <- mixedsort(sort(files))

        # get the predictions.
        predictions <- do.call(rbind, lapply(files, function(file) {
            temp <- read.csv(file, header = FALSE)
            names(temp) <- c(0:(length(temp) - 1))
            # attach relevant information.
            temp$model <- model
            temp$mode <- mode

            return(temp)
        }))

        # get the labels.
        input.file <- list.files(file.path("inputs", model, "production"),
                                 pattern = ".csv", full.names = TRUE)

        label <- scan(pipe(paste("cut -f1 -d,", input.file)))

        # combine.
        temp.2 <-  cbind(predictions, label)

        return(temp.2)
    }))

    return(data.set)
}))

# score.
data$score <- apply(data, 1, function(x) {
    preds <- x[1:(length(x) - 3)]
    label <- x['label']
    prediction <- names(preds)[which.max(preds)]

    if (prediction == label) {
        return(1)
    } else {
        return(0)
    }
})
