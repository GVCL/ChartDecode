
import sys
from Extract_DataTables.extractPie import extPie
from Extract_DataTables.extractLine import extLine
from Extract_DataTables.extractScatter import extScatterTypes
from Summary_Generation.ScattBubb import genrateSumm_SB
from Classifier_Models.classifyChart import classifyImage

# if __name__ == '__main__':
#     print("\033[1;32m INFO: This system works for Line, Pie, Simple Scatter, Bubbule, and Dot plots")
#     print("\u001b[31m \nWarning ....!!!! \n\n* The input chart image should have both X and Y axis labels(except for PIE chart)\nand graphical objects in canvas region for system to work sucessfully")
#     print("\n* Make sure graphical makers are represented using distinct \ncolor-pixels in multi-type chart images for system to work sucessfully")
#     print("\u001b[0m ________________________________________________________________________________________________________")
#     filename = input("Enter chart image name with its path : ")
#     chart_type = classifyImage(filename)
#     print("The chart image is classified as : ",chart_type)
#     if chart_type == 'pie' :
#         extPie(filename)
#     elif chart_type == 'line':
#         extLine(filename)
#     elif chart_type == 'scatter' :
#         extScatterTypes(filename)
#     else :
#         if chart_type == 'bar' :
#             sys.exit("\u001b[31m FAILED: This system doesn't handle bar charts and its variants")
#         sys.exit("\u001b[31m FAILED: The chart type can't be detected")


for i in range(1,16):
    # print(classifyImage('/Users/daggubatisirichandana/PycharmProjects/ChartDecode/SYNTHETIC_DATA/SCATTER/m'+str(i)+'.png'))
    print(i)
    extScatterTypes('/Users/daggubatisirichandana/PycharmProjects/ChartDecode/SYNTHETIC_DATA/DOT/dp'+str(i)+'.png')
