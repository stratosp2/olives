library("fpp3")
library("psych")
library("lubridate")
library("zoo")
library("reshape2")
library("tidyverse")
library("dplyr")
library("MASS")


rain <- read.csv("rain_zichni.csv")
temps <- read.csv("temperature_zichni.csv")
press <- read.csv("pressure_zichni.csv")
snow <- read.csv("snow_zichni.csv")
sunhrs <- read.csv("sunhours_zichni.csv")
wind <- read.csv("wind_zichni.csv")
clouds <- read.csv("clouds_zichni.csv")
#length(clouds$Date)

df_list <- list(rain, temps, press, snow, sunhrs, wind, clouds)   


df <- df_list %>% reduce(full_join, by='Date')
colnames(df) <- c("Date", "Rain", "Avg_Temp", "Max_Temp", "Min_Temp", "Pressure", "Snow", "Sun_Hours", "Max_Wind", "Avg_Wind",
                  "Avg_Gust", "Clouds")

tail(df)

write.csv(df,"Nea_Zichni_scrapped_data_full.csv")

df <-read.csv("Nea_Zichni_scrapped_data_full.csv")



df$Date <- as.Date(df$Date)

#Explore monthly first
df_monthly <- df%>%dplyr::select(!"X")

df_monthly$Month <- month(df_monthly$Date)
df_monthly %>% dplyr::select(c("Month","Rain", "Avg_Temp", "Min_Temp", "Max_Temp")) %>% group_by(Month)%>%summarise_all(., mean) -> means
colnames(means) <- c("Month","Avg Rain", "Avg Avg_Temp", "Avg Min_Temp", "Avg Max_Temp")

ts_mon_data <- df_monthly %>% as_tsibble(index = Date, key = NULL, regular = FALSE)



cols = length(unique(year(ts_mon_data$Date)))

mean_max_wind_3 <- ts_mon_data%>%filter(month(Date) == 3)%>%dplyr::pull(Max_Wind)%>%mean()
mean_avg_temp_4 <- ts_mon_data%>%filter(month(Date) == 4)%>%dplyr::pull(Avg_Temp)%>%mean()
mean_max_temp_6 <- ts_mon_data%>%filter(month(Date) == 6)%>%dplyr::pull(Max_Temp)%>%mean()
mean_clouds_8   <- ts_mon_data%>%filter(month(Date) == 8)%>%dplyr::pull(Clouds)%>%mean()
mean_avg_temp_9 <- ts_mon_data%>%filter(month(Date) == 9)%>%dplyr::pull(Avg_Temp)%>%mean()


ts_mon_data %>%mutate(Year = year(Date), Month = month(Date, label = TRUE)) %>%
  ggplot(aes(x = Month, y = Rain, fill = as.factor(Year))) +
  geom_col(position = "dodge") +
  labs(title = "Monthly Rainfall by Year", x = "Month", y = "Rain (mm)", fill = "Year") +
  theme_minimal()

ts_mon_data %>% mutate(Year = year(Date), Month = month(Date, label = TRUE)) %>%
  ggplot(aes(x = Month, y = Max_Wind, fill = as.factor(Year))) +
  geom_col(position = "dodge") +
  geom_point(aes(x = "Mar" , y = mean_max_wind_3, size = 2), shape = 18)+
  labs(title = "Monthly Max wind by Year", x = "Month", y = "Wind km/h", fill = "Year") +
  theme_minimal()

ts_mon_data %>% mutate(Year = year(Date), Month = month(Date, label = TRUE)) %>%
  ggplot(aes(x = Month, y = Avg_Temp, fill = as.factor(Year))) +
  geom_col(position = "dodge") +
  geom_point(aes(x = "Apr" , y = mean_avg_temp_4, size = 2), shape = 18)+
  geom_point(aes(x = "Sep" , y = mean_avg_temp_9, size = 2), shape = 18)+
  labs(title = "Monthly Avg temp by Year", x = "Month", y = "Temperature", fill = "Year") +
  theme_minimal()

ts_mon_data %>% mutate(Year = year(Date), Month = month(Date, label = TRUE)) %>%
  ggplot(aes(x = Month, y = Max_Temp, fill = as.factor(Year))) +
  geom_col(position = "dodge") +
  geom_point(aes(x = "Jun" , y = mean_max_temp_6, size = 2), shape = 18)+
  labs(title = "Monthly Max temp by Year", x = "Month", y = "Temperature", fill = "Year") +
  theme_minimal()

ts_mon_data %>% mutate(Year = year(Date), Month = month(Date, label = TRUE)) %>%
  ggplot(aes(x = Month, y = Clouds, fill = as.factor(Year))) +
  geom_col(position = "dodge") +
  geom_point(aes(x = "Aug" , y = mean_clouds_8, size = 2), shape = 18)+
  labs(title = "Monthly Clouds by Year", x = "Month", y = "Cloud hours", fill = "Year") +
  theme_minimal()

df$year = as.integer(year(ymd(df$Date)))

df$month = as.integer(month(ymd(df$Date)))

df$Spread = df$Max_Temp - df$Min_Temp
  
df%>%summary


df_yearly <- df%>%dplyr::select(!"X")

tail(df_yearly)


df_yearly_years <- df_yearly%>%dplyr::select("year", "Rain", "Avg_Temp", "Max_Temp", "Min_Temp", "Pressure", "Snow", "Sun_Hours", "Max_Wind", "Avg_Wind",
                                    "Avg_Gust", "Clouds")%>%group_by(year)%>%summarise_all(.,mean)

df_monthly = df_yearly%>%dplyr::select("month",  "Rain", "Avg_Temp", "Max_Temp", "Min_Temp", "Pressure", "Snow", "Sun_Hours", "Max_Wind", "Avg_Wind",
                                "Avg_Gust", "Clouds")%>%group_by(month)%>%summarise_all(.,mean)


ts_year_mean_data <- df_yearly_years %>% as_tsibble(index = year, key = NULL, regular = FALSE)
ts_mon_mean_data <- df_monthly %>% as_tsibble(index = month, key = NULL, regular = FALSE)

ts_year_mean_data %>%autoplot(Rain)

ts_yearly_tgt_data <- rbind(list(AvgTemp = ts_year_mean_data$Avg_Temp, MaxTemp = ts_year_mean_data$Max_Temp))

monthly_df <- ts_mon_data %>% 
  mutate(
    year = as.integer(str_sub(Date, 1, 4))
    #month = str_sub(Date, 6, 7)  # e.g., "1", "2"
  )

monthly_df%>%tail(5)


# Pivot the data wider by month
df_monthly_wide <- monthly_df%>%as_tibble() %>%
  dplyr::select(year, Month, Rain, Max_Temp, Avg_Temp, Min_Temp, Pressure, Snow, Sun_Hours, Max_Wind, Avg_Wind, Avg_Gust, Clouds) %>%
  pivot_wider(
    names_from = Month,
    values_from = c(Rain, Max_Temp, Avg_Temp, Min_Temp, Pressure, Snow, Sun_Hours, Max_Wind, Avg_Wind, Avg_Gust, Clouds),
    names_sep = "_"
  )


df_elies <-read.csv("elies.csv")
colnames(df_elies) <- c("index", "year", "trees", "olives", "oil", "ratio", "price" )
df_elies <- df_elies%>%dplyr::select(!c("index", "trees", "price"))%>%drop_na()

olive_m_data <-  merge(df_monthly_wide, df_elies, by='year')

tail(olive_m_data)

vars <- setdiff(names(olive_m_data), c("olives", "oil", "ratio", "Month", "year"))
formula <- as.formula(paste("olives ~", paste(vars, collapse = " + ")))

olive_means_mean_t <- olive_m_data%>%dplyr::select(!c(year, oil,ratio))

# Full model with all weather variables
#base_q_model <- lm(olives ~ 1 , data = olive_q_data)
#full_q_model <- lm(olives ~ Avg_Temp_Q3 + Clouds_Q2 +  Max_Temp_Q2   +
 #                    Sun_Hours_Q2 +  Sun_Hours_Q4, data = olive_q_data)
base_m_model <- lm(olives ~ 1 , data = olive_m_data)
full_model   <- lm(formula, data = olive_m_data)
full_m_olive_model <- lm(olives ~ Rain_3  + Min_Temp_1 + Min_Temp_2 + Max_Wind_3 + Pressure_3 + Min_Temp_4 + Avg_Temp_4 + 
                            Avg_Temp_6 + Max_Temp_6  + Min_Temp_6 + 
                            Sun_Hours_8 + Clouds_8 + Avg_Temp_9 + Min_Temp_9
                            , data = olive_m_data)

full_m_oil_model <- lm(oil ~ Rain_8  + Min_Temp_1 + Min_Temp_2 + Max_Wind_3 + Pressure_3 + Min_Temp_4 + Avg_Temp_4 + 
                           Avg_Temp_6 + Max_Temp_6  + Min_Temp_6 + 
                           Sun_Hours_8 + Clouds_8 + Avg_Temp_9 + Min_Temp_9
                         , data = olive_m_data)

# Stepwise model dplyr::dplyr::selection

step_m_olive_model <- stepAIC(full_m_olive_model, direction = "both", trace  = F)
step_m_oil_model <- stepAIC(full_m_oil_model, direction = "both", trace  = F)


# Summary of the best model
#summary(step_m_olive_model)
#summary(step_m_oil_model)

# Summary of the best model
final_model_olives_monthly <- lm (olives ~ Avg_Temp_4  + Max_Temp_6 + Max_Wind_3 +  Clouds_8 + Avg_Temp_9,
                           data = olive_m_data) 
final_model_oil_monthly <- lm (oil ~ Max_Wind_3 + Avg_Temp_4  + Max_Temp_6  + Clouds_8,
                                  data = olive_m_data) 

summary(final_model_olives_monthly)
summary(final_model_oil_monthly)



t_year = "2024"
max_wind_3  = olive_m_data %>%filter(year == t_year)%>%dplyr::select(Max_Wind_3)
avg_temp_4  = olive_m_data %>%filter(year == t_year)%>%dplyr::select(Avg_Temp_4)
max_temp_6  = olive_m_data %>%filter(year == t_year)%>%dplyr::select(Max_Temp_6)
clouds_8    = olive_m_data %>%filter(year == t_year)%>%dplyr::select(Clouds_8)
avg_temp_9  = olive_m_data %>%filter(year == t_year)%>%dplyr::select(Avg_Temp_9)

avg_temp_4*coefficients(summary(final_model_olives_monthly))[2] +
max_temp_6*coefficients(summary(final_model_olives_monthly))[3] + 
max_wind_3*coefficients(summary(final_model_olives_monthly))[4] +
clouds_8*coefficients(summary(final_model_olives_monthly))[5] +
avg_temp_9*coefficients(summary(final_model_olives_monthly))[6] +
coefficients(summary(final_model_olives_monthly))[1]


df <- olive_m_data %>%
  mutate(Predicted_Yield = predict(final_model_olives_monthly, newdata = olive_m_data))

df$Predicted_Yield <- pmax(0, df$Predicted_Yield)

df_comparison <- df %>%dplyr::select(year, olives, Predicted_Yield)
print(df_comparison)

df_long <- df_comparison %>%
  pivot_longer(cols = c("olives", "Predicted_Yield"),
               names_to = "Type",
               values_to = "olives")

ggplot(df_long, aes(x = year, y = olives, color = Type)) +
  geom_point(size = 3) +
  geom_point(size = 3) +
  labs(title = "Actual vs Predicted Olive Yield per Year",
       x = "Year", y = "Yield", color = "") +
  theme_minimal()


t_max_w_3 <- ts_mon_data%>%filter(Month == 3)%>%dplyr::select(Max_Wind)%>%tail(1)%>%pull(Max_Wind)
t_av_t_4 <- ts_mon_data%>%filter(Month == 4)%>%dplyr::select(Avg_Temp)%>%tail(1)%>%pull(Avg_Temp)
t_max_t_6 <- ts_mon_data%>%filter(Month == 6)%>%dplyr::pull(Max_Temp)%>%tail(1)
t_c_8 <- 16 #ts_mon_data%>%filter(Month == 8)%>%dplyr::pull(Clouds)%>%tail(1)
t_av_9 <- ts_mon_data%>%filter(Month == 9)%>%dplyr::pull(Avg_Temp)%>%tail(1)

t_av_t_4*coefficients(summary(final_model_olives_monthly))[2] +
  (t_max_t_6)*coefficients(summary(final_model_olives_monthly))[3] + 
  t_max_w_3*coefficients(summary(final_model_olives_monthly))[4] +
  (t_c_8)*coefficients(summary(final_model_olives_monthly))[5] + 
  t_av_9*coefficients(summary(final_model_olives_monthly))[6] + 
   coefficients(summary(final_model_olives_monthly))[1]

