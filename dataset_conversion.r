PATH <- "dataset/RData/TEP_Faulty_Testing.RData"
OUT_PATH <- "dataset/CSV/TEP_Faulty_Testing.csv"

data <- get(load(PATH))

write.csv(data, OUT_PATH)
# library(rhdf5)
# h5closeAll()
# h5createFile(OUT_PATH)
# h5write(data, OUT_PATH, "data")