function P = sph2ecef(lat_deg, lon_deg, r)
    lat = deg2rad(lat_deg(:));
    lon = deg2rad(lon_deg(:));
    r   = r(:);

    X = r .* cos(lat) .* cos(lon);
    Y = r .* cos(lat) .* sin(lon);
    Z = r .* sin(lat);

    P = [X Y Z];
end
