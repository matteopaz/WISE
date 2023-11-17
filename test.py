import astropy
from astroquery.ipac.irsa import Irsa
from astropy.coordinates import SkyCoord
import plotly.graph_objects as go
import astropy.units as u
import re

def get_coordinates(coordstring):
    hour_regex = re.compile(r'^[+-]?([0-9]{2,3}){1}\s+([0-9]{2}){1}\s+([0-9]{2}(\.[0-9]{1,})?){1}\s+[+-]{1}([0-9]{2,3}){1}\s+([0-9]{2}){1}\s+([0-9]{2}(\.[0-9]{1,})?){1}$')
    deg_regex = re.compile(r'^[+-]?([0-9]{1,}(.[0-9]{1,})?){1}\s+[+-]?([0-9]{1,}(.[0-9]{1,})?){1}$')

    is_hour = bool(hour_regex.search(coordstring))
    is_deg = bool(deg_regex.search(coordstring))

    if not is_hour and not is_deg:
        raise ValueError("Invalid coordinate string")

    if is_hour:
        print(coordstring, " is Hours")
        c = SkyCoord(coordstring, unit=(u.hourangle, u.deg))
    else:
        print(coordstring, " is Degrees")
        c = SkyCoord(coordstring, unit=(u.deg))

    return c



c1 = get_coordinates("07 28 44.79 -44 28 16.6")
c2 = get_coordinates("112.1866310 -44.4712826")

query1 = Irsa.query_region(c1, catalog='neowiser_p1bs_psd', spatial="Cone", radius=2.5 * u.arcsec)
query2 = Irsa.query_region(c2, catalog='neowiser_p1bs_psd', spatial="Cone", radius=2.5 * u.arcsec)

print(query1)
print(len(query1))
print(query2)
print(len(query2))


coords1 = query1["ra", "dec"]
coords2 = query2["ra", "dec"]

plot1 = go.Scatter(x=coords1["ra"], y=coords1["dec"], mode="markers")
plot2 = go.Scatter(x=coords2["ra"], y=coords2["dec"], mode="markers")

fig1 = go.Figure(plot1)
fig2 = go.Figure(plot2)

fig1.show()
fig2.show()
