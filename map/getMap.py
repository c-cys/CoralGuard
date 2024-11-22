import folium

def makeMap(latitude, longitude):    # latitude: 위도 / longitude: 경도
    global map
    map = folium.Map(location=[latitude, longitude], zoom_start=5)

def drawCircle(lat, long, r, colorLine, colorIn):
    folium.Circle(
        location=[lat, long],
        radius=r,
        color=colorLine,
        fill_color=colorIn,
        popup='Circle popup',
        tooltip='Circle tooltip'
    ).add_to(map)