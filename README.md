# FullProto3

import sys
import os
import busio
import digitalio
import board
import time
import csv
import pandas as pd
import adafruit_mcp3xxx.mcp3008 as MCP
from adafruit_mcp3xxx.analog_in import AnalogIn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import adafruit_character_lcd.character_lcd as characterlcd

lcd_columns = 16
lcd_rows = 2

lcd_rs = digitalio.DigitalInOut(board.D22)
lcd_en = digitalio.DigitalInOut(board.D17)
lcd_d4 = digitalio.DigitalInOut(board.D25)
lcd_d5 = digitalio.DigitalInOut(board.D24)
lcd_d6 = digitalio.DigitalInOut(board.D23)
lcd_d7 = digitalio.DigitalInOut(board.D18)

lcd = characterlcd.Character_LCD_Mono(lcd_rs, lcd_en, lcd_d4, lcd_d5, lcd_d6, lcd_d7, lcd_columns, lcd_rows)
spi = busio.SPI(clock=board.SCK, MISO=board.MISO, MOSI=board.MOSI)
cs = digitalio.DigitalInOut(board.D4)
mcp = MCP.MCP3008(spi, cs)

lcd.message = 'Watch Val1 at Rest:'
time.sleep(2)
watch_time = 30
watch_start = time.time()
while (time.time() - watch_start) < watch_time:
    lcd.clear()
    Val1 = AnalogIn(mcp, MCP.P0)
    lcd.message = 'Val1: %s'%Val1.value
    
Thresh = int(input('Threshold value greater than greatest Val1: '))

os.remove('Dead.csv')

max_time = int(input('Seconds you want to train OPEN: '))
start_time = time.time()
  
while (time.time() - start_time) < max_time:
        Val1 = AnalogIn(mcp, MCP.P0)
        Val2 = AnalogIn(mcp, MCP.P1)
        Val3 = (Val1.value + Val2.value)/2
        Val4 = Val1.value + Val2.value
        Val5 = Val1.value - Val2.value
        lcd.message = 'Val: %s'%Val1.value
        
        if Val1.value > Thresh:
            Pos = "open"
            with open("Dead.csv","a") as f:
                writer = csv.writer(f,delimiter=",")
                writer.writerow([Pos,Val1.value,Val2.value,Val3,Val4,Val5])
        
        if Val1.value < Thresh:
            Pos = "rest"
            with open("Dead.csv","a") as f:
                writer = csv.writer(f,delimiter=",")
                writer.writerow([Pos,Val1.value,Val2.value,Val3,Val4,Val5])

max_time = int(input('Seconds you want to train CLOSE: '))
start_time = time.time()

while (time.time() - start_time) < max_time:
        Val1 = AnalogIn(mcp, MCP.P0)
        Val2 = AnalogIn(mcp, MCP.P1)
        Val3 = (Val1.value + Val2.value)/2
        Val4 = Val1.value + Val2.value
        Val5 = Val1.value - Val2.value
        lcd.message = 'Val: %s'%Val1.value
        
        if Val1.value > Thresh:
            Pos = "close"
            with open("Dead.csv","a") as f:
                writer = csv.writer(f,delimiter=",")
                writer.writerow([Pos,Val1.value,Val2.value,Val3,Val4,Val5])
        
        if Val1.value < Thresh:
            Pos = "rest"
            with open("Dead.csv","a") as f:
                writer = csv.writer(f,delimiter=",")
                writer.writerow([Pos,Val1.value,Val2.value,Val3,Val4,Val5])

names = ['Position', 'Val1', 'Val2', 'Val3', 'Val4', 'Val5']
dataset = pd.read_csv('Dead.csv', names=names)

X = dataset.iloc[:, 1:5].values
y = dataset.iloc[:, 0].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

lda = LDA(n_components=1)
X_train = lda.fit_transform(X_train, y_train)
X_test = lda.transform(X_test)

classifier = RandomForestClassifier(max_depth=2, random_state=0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print(cm)
print('Accuracy' + str(accuracy_score(y_test, y_pred)))

lcd.message = 'Accuracy' + str(accuracy_score(y_test, y_pred))
time.sleep(2)
lcd.message = '  Is Acc Good?   ' + 'Yes No'
goon = input('Is the accuracy goodnuff Y/N?: ')
if goon == 'N':
    sys.exit()
    
if goon == 'Y':
    
    names = ['Position', 'Val1', 'Val2', 'Val3', 'Val4', 'Val5']
    dataset = pd.read_csv('Dead.csv', names=names)
    
    X_train = dataset.iloc[:, 1:5].values
    y_train = dataset.iloc[:, 0].values
    
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    
    lda = LDA(n_components=2)
    X_train = lda.fit_transform(X_train, y_train)
    
    classifier = RandomForestClassifier(max_depth=2, random_state=0)
    classifier.fit(X_train, y_train)
    
    os.remove('Active.csv')
    
    while True:
        lcd.clear()
        Val1 = AnalogIn(mcp, MCP.P0)
        Val2 = AnalogIn(mcp, MCP.P1)
        Val3 = (Val1.value + Val2.value)/2
        Val4 = Val1.value + Val2.value
        Val5 = Val1.value - Val2.value
        
        newnames = ['Val1', 'Val2', 'Val3', 'Val4', 'Val5']
        
        with open("Active.csv","a") as f:
            writer = csv.writer(f,delimiter=",")
            writer.writerow([Val1.value,Val2.value,Val3])
        
        ACsv = pd.read_csv("Active.csv", names=newnames)
        
        X_test = ACsv.iloc[:, 0:4]
    
        X_test = sc.transform(X_test)
        X_test = lda.transform(X_test)
    
        y_pred = classifier.predict(X_test)
        lcd.message = str(y_pred)
    
        print(y_pred)
        os.remove('Active.csv')
