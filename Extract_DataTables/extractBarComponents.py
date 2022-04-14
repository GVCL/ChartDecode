from utils import *

Path = "/Users/daggubatisirichandana/PycharmProjects/chart_percept/BCA-App/static/data/Test/"
image_name = 'vertical_stacked_bar/sb02/sb02.png'
# filename = Path+str(image_name)+".png"
filename = Path + image_name
chart_type = 'bar'#classifyImage(Path+str(image_name)+".png")
# graph_img = cv2.imread("/Users/daggubatisirichandana/PycharmProjects/chart_percept/Data_Extraction/Generated_data/vertical_grouped_bar/gb02/gb02.png")
graph_img = cv2.imread("/Users/daggubatisirichandana/PycharmProjects/chart_percept/GVCL.github.io/static/data/Test/Histogram/hist6/hist6.png")


img = graph_img.copy()
chart_dict, IS_MULTI_CHART, legend_colors, legend_names, _= extractCanvaLeg(img,chart_type)
img = graph_img.copy()
title, y_title, ybox_centers, Ylabel, x_title, xbox_centers, Xlabel = extractLablTitl(img,chart_dict,IS_MULTI_CHART)
# chart_dict['legend'] = {'x': 677, 'y': 40, 'w': 103, 'h': 60}
# chart_dict['title'] = {'x': 265, 'y': 12, 'w': 198, 'h': 25}
img = graph_img.copy()
cv2.imwrite("/Users/daggubatisirichandana/PycharmProjects/chart_percept/LINE_PIE/ThesisExpt/CCE_1.png",viewSegChart(img,chart_dict))

cv2.waitKey(0)


# {'canvas': {'x': 77, 'y': 58, 'w': 577, 'h': 533}, 'y-title': {'x': 11, 'y': 285, 'w': 18, 'h': 50}, 'y-labels': {'x': 34, 'y': 114, 'w': 11, 'h': 245}, 'title': {'x': 719, 'y': 39, 'w': 49, 'h': 18}, 'x-labels': {'x': 98, 'y': 587, 'w': 534, 'h': 22}, 'x-title': {'x': 338, 'y': 607, 'w': 55, 'h': 22}} False [[0, 0, 0]] ['_']
