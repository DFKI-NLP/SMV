from fleiss_kappa_thermostat_class import fleiss_kappa

datasets_agnews = ["AGNews_G0_Anno0.csv", "AGNews_G0_Anno1.csv", "AGNews_G0_Anno2.csv", "AGNews_G0_Anno3.csv",
                   "AGNews_G1_Anno4.csv", "AGNews_G1_Anno5.csv", "AGNews_G1_Anno6.csv",
                   "AGNews_G2_Anno7.csv", "AGNews_G2_Anno8.csv", "AGNews_G2_Anno9.csv",
                   "AGNews_G3_Anno0.csv", "AGNews_G3_Anno2.csv"
                   ]
col_names = ["idx", "sort_idx", "text", "explanation", "simulation", "helpful", "easy"]
filter_words_agnews = ["Sports", "Sci/Tech", "World", "Business"]

fleiss_agnews = fleiss_kappa()
score = fleiss_agnews.doit(datasets_agnews, col_names, filter_words_agnews)
print(f"ag_news fleiss_kappa score: {score}")


datasets_imdb = ["IMDb_G0_Anno0.csv", "IMDb_G0_Anno1.csv", "IMDb_G0_Anno2.csv", "IMDb_G0_Anno3.csv",
                 "IMDb_G1_Anno4.csv", "IMDb_G1_Anno5.csv", "IMDb_G1_Anno6.csv",
                 "IMDb_G2_Anno7.csv", "IMDb_G2_Anno8.csv", "IMDb_G2_Anno9.csv",
                 "IMDb_G3_Anno0.csv", "IMDb_G3_Anno2.csv"
                 ]
filter_words_imdb = ["Positive sentiment","Negative sentiment"]
fleiss_imdb = fleiss_kappa()
score = fleiss_imdb.doit(datasets_imdb, col_names, filter_words_imdb)
print(f"imdb fleiss_kappa score: {score}")

print()
