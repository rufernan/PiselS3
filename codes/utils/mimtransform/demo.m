% modify this to point where you unzipped:
base_dir = 'c:\work\other\fx\mimtransform\';

filename = 'NED_40216094.tif';
filepath = fullfile(base_dir, filename);
lat0 = 39.949493408;
lon0 = -105.194267273;

%%
[grd, R, bb] = geotiffread(filepath);

mstruct = struct('mapprojection','none');
mstruct2 = defaultm('utm');
mstruct2.zone = utmzone(fliplr(mean(bb)));
mstruct2.geoid = almanac('earth','grs80','meters');
mstruct2 = defaultm(mstruct2);

%%
[grd2, xl, yl] = mimtransform (grd, R, mstruct, mstruct2);
[x0, y0] = mfwdtran(mstruct2, lat0, lon0);

xd = linspace(xl(1), xl(2), size(grd2,2));
yd = linspace(yl(1), yl(2), size(grd2,1));
z0 = interp2(xd, yd, grd2, x0, y0);

%%
%z0=0;
figure
subplot(1,2,1)
  imagesc(bb(:,1), bb(:,2), flipud(grd)-z0)
  xlabel('Longitude (degrees)')
  ylabel('Latitude (degrees)')
subplot(1,2,2)
  imagesc(xl-x0, yl-y0, grd2-z0)
  xlabel('East (m)')
  ylabel('North (m)')
for i=1:2
  subplot(1,2,i)
  set(gca, 'YDir','normal')
  axis image
  grid on
end
subplot(1,2,1),  xlabel(colorbar('SouthOutside'), '(meters)')

