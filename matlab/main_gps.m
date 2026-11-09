%% GPS receiver positioning: input data, least-squares solution, sensitivity analysis
clear; clc; close all;

projRoot = fileparts(fileparts(mfilename('fullpath')));
outDir   = fullfile(projRoot, 'results');
if ~exist(outDir, 'dir'); mkdir(outDir); end
logFile = fullfile(outDir, 'log.txt');
if exist(logFile, 'file'); delete(logFile); end
diary(logFile); diary on;

format long g

set(groot, 'defaultFigureColor', 'w');
set(groot, 'defaultAxesColor', 'w');
set(groot, 'defaultAxesXColor', 'k');
set(groot, 'defaultAxesYColor', 'k');
set(groot, 'defaultAxesZColor', 'k');
set(groot, 'defaultTextColor', 'k');
set(groot, 'defaultAxesGridColor', [0.15 0.15 0.15]);
set(groot, 'defaultLegendTextColor', 'k');
set(groot, 'defaultLegendColor', 'w');
set(groot, 'defaultLegendEdgeColor', [0.4 0.4 0.4]);

%% Constants
R = 6378137;         % reference sphere radius, sea level [m]
c = 299792458;       % signal speed [m/s]

fprintf('--- Setup ---\n');
fprintf('R = %.0f m, c = %.0f m/s\n\n', R, c);


%% 1) INPUT DATA
dataFile = fullfile(projRoot, 'data', 'dane_45.csv');
T = readtable(dataFile, 'Delimiter', ';', 'VariableNamingRule', 'preserve', 'Encoding', 'UTF-8');
disp(T);

name = T.("Nr satelity");
lat  = T.("Szerokość dziesiętna [°]");
lon  = T.("Długość dziesiętna [°]");
h    = T.("Wysokość n.p.m. [m]");
t    = T.("Czas nadejścia sygnału GPS [s]");
N    = height(T);

fprintf('Number of satellites N = %d\n', N);
fprintf('Altitude  h : min = %.3f km, max = %.3f km\n', min(h)/1e3, max(h)/1e3);
fprintf('Time      t : min = %.9f s, max = %.9f s (spread %.3f ms)\n\n', ...
    min(t), max(t), 1e3*(max(t)-min(t)));

% Geocentric radius of a satellite: sea level R plus its altitude h
rSat = R + h;
Sat  = sph2ecef(lat, lon, rSat);

fprintf('--- Satellite positions (ECEF) [m] ---\n');
for i = 1:N
    fprintf('Sat %-4s : X=%14.3f  Y=%14.3f  Z=%14.3f\n', name{i}, Sat(i,1), Sat(i,2), Sat(i,3));
end
fprintf('\n');

d = c * t;

dMin = rSat - R;
dMax = sqrt(rSat.^2 - R^2);
okRange = d > dMin & d < dMax;

fprintf('--- Ranges d_i = c*t_i and plausibility check [m] ---\n');
fprintf('%-6s %16s %16s %16s   %s\n', 'Sat', 'd_i', 'min (zenith)', 'max (horizon)', 'above horizon');
for i = 1:N
    fprintf('%-6s %16.3f %16.3f %16.3f   %s\n', name{i}, d(i), dMin(i), dMax(i), string(okRange(i)));
end
fprintf('All %d satellites inside the plausible range: %s\n\n', N, string(all(okRange)));

% Figure: sub-satellite points on the world map
fig = figure('Visible','off', 'Color','w', 'Position',[100 100 1000 560]);
try
    cl = load('coastlines'); 
    plot(cl.coastlon, cl.coastlat, 'Color', [0.6 0.6 0.6]); hold on;
catch
    hold on;
end
scatter(lon, lat, 70, 'filled', 'MarkerFaceColor', [0.85 0.1 0.1]);
text(lon+2, lat+2, name, 'FontSize', 9, 'Color', 'k');
ax = gca; set(ax, 'Color','w', 'XColor','k', 'YColor','k');
xlim([-180 180]); ylim([-90 90]); grid on; box on;
xticks(-180:60:180); yticks(-90:30:90);
xlabel('Longitude [deg]', 'Color','k'); ylabel('Latitude [deg]', 'Color','k');
title('Sub-satellite points of the eight GPS satellites', 'Color','k');
drawnow;
exportgraphics(fig, fullfile(outDir,'data_groundtrack.png'), 'Resolution',150, 'BackgroundColor','white');

% Figure: overview of the raw input columns
fig = figure('Visible','off', 'Color','w', 'Position',[100 100 1100 420]);
tl = tiledlayout(1, 3, 'TileSpacing','compact', 'Padding','compact');

nexttile; bar(h/1e3, 'FaceColor', [0.2 0.45 0.75]);
set(gca,'XTick',1:N,'XTickLabel',name,'Color','w','XColor','k','YColor','k');
ylabel('h [km]','Color','k'); title('Satellite altitude','Color','k'); grid on;
ylim([19800 20500]);

nexttile; bar(t*1e3, 'FaceColor', [0.85 0.55 0.1]);
set(gca,'XTick',1:N,'XTickLabel',name,'Color','w','XColor','k','YColor','k');
ylabel('t [ms]','Color','k'); title('Signal arrival time','Color','k'); grid on;
ylim([66 79]);

nexttile; bar(d/1e3, 'FaceColor', [0.3 0.6 0.3]);
set(gca,'XTick',1:N,'XTickLabel',name,'Color','w','XColor','k','YColor','k');
ylabel('d = c\cdot t [km]','Color','k'); title('Measured range','Color','k'); grid on;
ylim([19500 24000]);

title(tl, 'Input data read from dane\_45.csv', 'Color','k');
drawnow;
exportgraphics(fig, fullfile(outDir,'data_overview.png'), 'Resolution',150, 'BackgroundColor','white');


%% 2) MODEL AND OPTIMIZATION PROBLEM
% System of equations (N equations, 3 unknowns x, y, z):
%     || [x,y,z] - Sat_i || = d_i ,  i = 1..N
% Unconstrained least-squares formulation:
%     min_{x,y,z}  sum_i ( || [x,y,z] - Sat_i || - d_i )^2

modelFun = @(p) gps_model(p, Sat, d);                % [F, J], for lsqnonlin
resFun   = @(p) gps_residuals(p, Sat, d);            % F only, for lm_solve
jacFun   = @(p) gps_jacobian(p, Sat);

% Starting point: on the reference sphere, under the mean direction of the
% satellites, since the receiver should be somewhere near the surface.
dirMean = mean(Sat ./ vecnorm(Sat,2,2), 1);
dirMean = dirMean / norm(dirMean);
x0_base = R * dirMean;

fprintf('--- Base starting point (Earth surface, under satellite centroid) ---\n');
fprintf('x0 = [%.3f, %.3f, %.3f]\n\n', x0_base(1), x0_base(2), x0_base(3));


%% 3) SOLUTION - LEVENBERG-MARQUARDT
fprintf('------------------------------------------------------------\n');
fprintf(' 3a) lsqnonlin (Optimization Toolbox), algorithm = levenberg-marquardt\n');
fprintf('------------------------------------------------------------\n');

optsLsq = optimoptions('lsqnonlin', ...
    'Algorithm', 'levenberg-marquardt', ...
    'SpecifyObjectiveGradient', true, ...
    'FunctionTolerance', 1e-12, ...
    'StepTolerance', 1e-12, ...
    'MaxIterations', 500, ...
    'Display', 'iter', ...
    'OutputFcn', @lsq_history);

lsqHistory = zeros(0,2);
[p_lsq, resnorm_lsq, residual_lsq, exitflag_lsq, output_lsq] = ...
    lsqnonlin(modelFun, x0_base(:), [], [], optsLsq);
p_lsq = p_lsq(:)';

fprintf('\n--- Reference solution (lsqnonlin, tol = 1e-12, base x0) ---\n');
fprintf('x = %.6f\ny = %.6f\nz = %.6f\n', p_lsq(1), p_lsq(2), p_lsq(3));
fprintf('Objective value (resnorm) = %.6e\n', resnorm_lsq);
fprintf('Residual norm = %.6e m\n', norm(residual_lsq));
fprintf('Iterations = %d, function evaluations = %d, exitflag = %d\n', ...
    output_lsq.iterations, output_lsq.funcCount, exitflag_lsq);
fprintf('First-order optimality = %.6e\n\n', output_lsq.firstorderopt);

p_ref = p_lsq;   % reference for all the comparisons below

fprintf('------------------------------------------------------------\n');
fprintf(' 3b) Own Levenberg-Marquardt implementation (cross-check)\n');
fprintf('------------------------------------------------------------\n');

optsOwn = struct('Display', 'iter', 'FunctionTolerance', 1e-12, ...
    'StepTolerance', 1e-12, 'MaxIterations', 500, 'lambda0', 1e-3);
[p_own, resnorm_own, residual_own, exitflag_own, output_own] = ...
    lm_solve(resFun, jacFun, x0_base, optsOwn);
p_own = p_own(:)';

fprintf('\n--- Solution (lm_solve) ---\n');
fprintf('x = %.6f\ny = %.6f\nz = %.6f\n', p_own(1), p_own(2), p_own(3));
fprintf('Objective value (resnorm) = %.6e\n', resnorm_own);
fprintf('Residual norm = %.6e m\n', norm(residual_own));
fprintf('Iterations = %d, function evaluations = %d, exitflag = %d\n', ...
    output_own.iterations, output_own.funcCount, exitflag_own);
fprintf('Distance lsqnonlin <-> own implementation: %.6e m\n\n', norm(p_lsq(:) - p_own(:)));

% Figure: convergence of both solvers
fig = figure('Visible','off', 'Color','w', 'Position',[100 100 950 550]);
semilogy(lsqHistory(:,1), max(lsqHistory(:,2), realmin), '-o', 'LineWidth', 1.6, 'MarkerSize', 7); hold on;
semilogy(output_own.history(:,1), max(2*output_own.history(:,2), realmin), '--s', 'LineWidth', 1.6, 'MarkerSize', 7);
set(gca, 'Color','w', 'XColor','k', 'YColor','k'); grid on;
xticks(0:max(lsqHistory(end,1), output_own.history(end,1)));
xlabel('Iteration', 'Color','k');
ylabel('Sum of squared residuals  \Sigma F_i^2  [m^2]', 'Color','k');
title('Convergence of the Levenberg-Marquardt method', 'Color','k');
legend({'lsqnonlin (Optimization Toolbox)', 'lm\_solve (own implementation)'}, 'TextColor','k', 'Location','northeast');
drawnow;
exportgraphics(fig, fullfile(outDir,'convergence.png'), 'Resolution',150, 'BackgroundColor','white');

% Figure: per-satellite residuals at the optimum
fig = figure('Visible','off', 'Color','w', 'Position',[100 100 950 450]);
bar(residual_lsq*1e3, 'FaceColor', [0.45 0.3 0.7]);
set(gca, 'XTick', 1:N, 'XTickLabel', name, 'Color','w', 'XColor','k', 'YColor','k'); grid on;
xlabel('Satellite', 'Color','k');
ylabel('F_i(p^*) [mm]', 'Color','k');
title('Residuals at the optimum (model range minus measured range)', 'Color','k');
drawnow;
exportgraphics(fig, fullfile(outDir,'residuals.png'), 'Resolution',150, 'BackgroundColor','white');


%% 4) RECEIVER POSITION IN GEOGRAPHIC COORDINATES
[lat_sol, lon_sol, h_sol] = ecef2sph(p_ref, R);
fprintf('--- Receiver position, geographic coordinates ---\n');
fprintf('Latitude  = %.8f deg\n', lat_sol);
fprintf('Longitude = %.8f deg\n', lon_sol);
fprintf('Height above sea level = %.3f m\n', h_sol);
fprintf('Google Maps: https://www.google.com/maps?q=%.8f,%.8f\n\n', lat_sol, lon_sol);

% Figures: the result drawn on real OpenStreetMap tiles
osmName = 'osm_tiles';
try
    try, removeCustomBasemap(osmName); catch, end
    addCustomBasemap(osmName, 'https://a.tile.openstreetmap.org/${z}/${x}/${y}.png', ...
        'Attribution', '(c) OpenStreetMap contributors', 'MaxZoomLevel', 18);
catch err
    warning('Could not register the OSM basemap (%s); falling back to ''streets''.', err.message);
    osmName = 'streets';
end

try
    % close-up
    fig = figure('Visible','off', 'Color','w', 'Position',[100 100 950 750]);
    geoplot(lat_sol, lon_sol, 'v', 'MarkerSize', 13, 'MarkerFaceColor', [0.85 0.1 0.1], 'MarkerEdgeColor', 'k');
    geobasemap(osmName);
    geolimits(lat_sol + [-0.006 0.006], lon_sol + [-0.009 0.009]);
    title(sprintf('Computed position: %.6f N, %.6f E (OpenStreetMap)', lat_sol, lon_sol), 'Color','k');
    drawnow; pause(6); drawnow;
    exportgraphics(fig, fullfile(outDir,'map_osm.png'), 'Resolution',150, 'BackgroundColor','white');

    % regional context
    fig = figure('Visible','off', 'Color','w', 'Position',[100 100 950 750]);
    geoplot(lat_sol, lon_sol, 'v', 'MarkerSize', 13, 'MarkerFaceColor', [0.85 0.1 0.1], 'MarkerEdgeColor', 'k');
    geobasemap(osmName);
    geolimits(lat_sol + [-0.55 0.55], lon_sol + [-0.85 0.85]);
    title('Regional context of the computed position', 'Color','k');
    drawnow; pause(6); drawnow;
    exportgraphics(fig, fullfile(outDir,'map_context.png'), 'Resolution',150, 'BackgroundColor','white');
    fprintf('Map figures written (basemap: %s).\n\n', osmName);
catch err
    warning('Map figures skipped (no internet access for the map tiles?): %s', err.message);
end


%% 5) OBSERVATION GEOMETRY: ELEVATION, AZIMUTH, DOP
fprintf('------------------------------------------------------------\n');
fprintf(' 5) OBSERVATION GEOMETRY SEEN FROM THE RECEIVER\n');
fprintf('------------------------------------------------------------\n');

[dop, elev, azim] = dop_metrics(Sat, p_ref, lat_sol, lon_sol);

fprintf('%-6s %12s %12s\n', 'Sat', 'elev [deg]', 'azim [deg]');
for i = 1:N
    fprintf('%-6s %12.2f %12.2f\n', name{i}, elev(i), azim(i));
end
fprintf('\nDilution of precision (position only, 3 unknowns):\n');
fprintf('  HDOP = %.4f, VDOP = %.4f, PDOP = %.4f\n', dop.HDOP, dop.VDOP, dop.PDOP);
fprintf('Including a receiver clock bias (4 unknowns):\n');
fprintf('  GDOP = %.4f, TDOP = %.4f\n\n', dop.GDOP, dop.TDOP);

%% 6a) SENSITIVITY: STARTING POINT
fprintf('------------------------------------------------------------\n');
fprintf(' 6a) EFFECT OF THE STARTING POINT\n');
fprintf('------------------------------------------------------------\n');

startNames = { ...
    'Base (under centroid, R)', ...
    'Earth center', ...
    'Antipode (surface)', ...
    'Base + 1000 km (random direction)', ...
    'Far away: 10*R along base direction' };

startVals = { ...
    x0_base, ...
    [1 1 1], ...                 % practically (0,0,0), avoids a zero Jacobian
    -x0_base, ...
    x0_base + 1e6*[1 -1 0.5]/norm([1 -1 0.5]), ...
    10*R*dirMean };

optsLsqQuiet = optimoptions(optsLsq, 'Display', 'off', 'OutputFcn', []);

fprintf('%-38s | %14s %14s %14s | %10s | %6s %6s | %12s\n', ...
    'Starting point', 'x', 'y', 'z', '||res||[m]', 'iter', 'exfl', 'dist2ref[m]');
for k = 1:numel(startNames)
    [pk, ~, resk, efk, outk] = lsqnonlin(modelFun, startVals{k}(:), [], [], optsLsqQuiet);
    fprintf('%-38s | %14.3f %14.3f %14.3f | %10.3e | %6d %6d | %12.3e\n', ...
        startNames{k}, pk(1), pk(2), pk(3), norm(resk), outk.iterations, efk, norm(pk(:)-p_ref(:)));
end

% Figure: basin of attraction over a grid of starting points
u = dirMean(:);
v = null(u');  v = v(:,1);                     % a unit vector perpendicular to u
radii  = logspace(log10(0.05*R), log10(15*R), 45);
angles = linspace(0, 180, 45);
errGrid = zeros(numel(angles), numel(radii));
for ia = 1:numel(angles)
    a    = deg2rad(angles(ia));
    dirA = cos(a)*u + sin(a)*v;
    for ir = 1:numel(radii)
        pk = lsqnonlin(modelFun, radii(ir)*dirA, [], [], optsLsqQuiet);
        errGrid(ia, ir) = norm(pk(:) - p_ref(:));
    end
end
convFrac = mean(errGrid(:) < 1);
fprintf('\nBasin of attraction scan: %d starting points, %.1f%% converged to the reference solution (< 1 m).\n\n', ...
    numel(errGrid), 100*convFrac);

fig = figure('Visible','off', 'Color','w', 'Position',[100 100 950 600]);
imagesc(log10(radii/R), angles, double(errGrid >= 1));   % 0 = converged, 1 = failed
set(gca, 'YDir','normal', 'Color','w', 'XColor','k', 'YColor','k');
colormap([0.25 0.55 0.85; 0.92 0.72 0.15]);
clim([0 1]);
hold on;
hOk   = patch(NaN, NaN, [0.25 0.55 0.85]);
hFail = patch(NaN, NaN, [0.92 0.72 0.15]);
hP0   = plot(0, 0, 'kp', 'MarkerSize', 16, 'MarkerFaceColor', 'w', 'LineWidth', 1.2);
legend([hOk hFail hP0], ...
    {'converged to p_{ref} (< 1 m)', 'converged to another stationary point', 'base starting point p_0'}, ...
    'Location', 'northwest', 'TextColor', 'k');
xlabel('Starting-point radius  log_{10}(r / R)', 'Color','k');
ylabel('Angle from the base direction [deg]', 'Color','k');
title(sprintf('Basin of attraction of the reference solution (%d starting points, %.1f%% converged)', ...
    numel(errGrid), 100*convFrac), 'Color','k');
drawnow;
exportgraphics(fig, fullfile(outDir,'start_basin.png'), 'Resolution',150, 'BackgroundColor','white');


%% 6b) SENSITIVITY: STOP TEST TOLERANCE
fprintf('------------------------------------------------------------\n');
fprintf(' 6b) EFFECT OF STOP TEST TOLERANCE (FunctionTolerance = StepTolerance)\n');
fprintf('------------------------------------------------------------\n');

tols    = [1e-2 1e-4 1e-6 1e-8 1e-10 1e-12 1e-14];
tolIter = zeros(size(tols));
tolErr  = zeros(size(tols));

fprintf('%10s | %14s %14s %14s | %10s | %6s %6s | %12s\n', ...
    'tol', 'x', 'y', 'z', '||res||[m]', 'iter', 'exfl', 'dist2ref[m]');
for k = 1:numel(tols)
    optsT = optimoptions(optsLsqQuiet, 'FunctionTolerance', tols(k), 'StepTolerance', tols(k));
    [pk, ~, resk, efk, outk] = lsqnonlin(modelFun, x0_base(:), [], [], optsT);
    tolIter(k) = outk.iterations;
    tolErr(k)  = norm(pk(:) - p_ref(:));
    fprintf('%10.0e | %14.3f %14.3f %14.3f | %10.3e | %6d %6d | %12.3e\n', ...
        tols(k), pk(1), pk(2), pk(3), norm(resk), outk.iterations, efk, tolErr(k));
end

fig = figure('Visible','off', 'Color','w', 'Position',[100 100 950 520]);
yyaxis left
semilogx(tols, max(tolErr, 1e-12), '-o', 'LineWidth', 1.6, 'MarkerSize', 7);
set(gca, 'YScale','log'); ylabel('Distance from the reference solution [m]', 'Color','k');
yyaxis right
semilogx(tols, tolIter, '--s', 'LineWidth', 1.6, 'MarkerSize', 7);
ylabel('Number of iterations', 'Color','k'); ylim([0 max(tolIter)+2]);
set(gca, 'Color','w', 'XColor','k'); grid on;
xlabel('FunctionTolerance = StepTolerance', 'Color','k');
title('Effect of the STOP test tolerance on accuracy and cost', 'Color','k');
legend({'distance from the reference solution (left axis)', 'number of iterations (right axis)'}, ...
    'Location', 'west', 'TextColor', 'k');
drawnow;
exportgraphics(fig, fullfile(outDir,'tolerance.png'), 'Resolution',150, 'BackgroundColor','white');


%% 6c) SENSITIVITY: NOISE IN THE SIGNAL ARRIVAL TIMES
fprintf('\n------------------------------------------------------------\n');
fprintf(' 6c) EFFECT OF NOISE IN SIGNAL ARRIVAL TIMES\n');
fprintf('------------------------------------------------------------\n');

rng(45, 'twister');   % reproducible results (index number 45)
sigma_t_list = [0, 1e-9, 1e-8, 1e-7, 1e-6];   % seconds
nMC = 200;                                    % trials per noise level

errMean = zeros(size(sigma_t_list));
errStd  = zeros(size(sigma_t_list));
errMax  = zeros(size(sigma_t_list));
mcENU   = [];        % ENU scatter kept for sigma_t = 1e-8 s

fprintf('%12s | %12s | %10s %10s %10s\n', 'sigma_t [s]', 'sigma_d [m]', 'mean err[m]', 'std err[m]', 'max err[m]');
for k = 1:numel(sigma_t_list)
    sig_t = sigma_t_list(k);
    errs  = zeros(nMC,1);
    enu   = zeros(nMC,3);
    for m = 1:nMC
        d_noisy = c * (t + sig_t*randn(N,1));
        pk = lsqnonlin(@(p) gps_model(p, Sat, d_noisy), x0_base(:), [], [], optsLsqQuiet);
        errs(m)  = norm(pk(:) - p_ref(:));
        enu(m,:) = ecef2enu((pk(:) - p_ref(:))', lat_sol, lon_sol);
    end
    errMean(k) = mean(errs);
    errStd(k)  = std(errs);
    errMax(k)  = max(errs);
    if sig_t == 1e-8, mcENU = enu; end
    fprintf('%12.1e | %12.3f | %10.3f %10.3f %10.3f\n', sig_t, c*sig_t, errMean(k), errStd(k), errMax(k));
end

fig = figure('Visible','off', 'Color','w', 'Position',[100 100 900 550]);
errorbar(sigma_t_list, errMean, errStd, '-o', 'LineWidth', 1.5, 'MarkerSize', 6);
set(gca, 'XScale','log', 'YScale','log', 'Color','w', 'XColor','k', 'YColor','k'); grid on;
xlabel('Time noise std. dev. \sigma_t [s]', 'Color','k');
ylabel('Position error vs. reference solution [m]', 'Color','k');
title('Effect of signal timing noise on the estimated position (MC, n=200)', 'Color','k');
drawnow;
exportgraphics(fig, fullfile(outDir,'time_noise_sensitivity.png'), 'Resolution',150, 'BackgroundColor','white');

% Figure: horizontal scatter + 95% confidence ellipse (sigma_t = 1e-8 s)
C = cov(mcENU(:,1:2));
[V, D] = eig(C);
th  = linspace(0, 2*pi, 200);
ell = (V * sqrt(D) * sqrt(5.991) * [cos(th); sin(th)])';   % chi2(2, 0.95) = 5.991
semiMajor95 = sqrt(5.991 * max(diag(D)));

fig = figure('Visible','off', 'Color','w', 'Position',[100 100 760 720]);
scatter(mcENU(:,1), mcENU(:,2), 22, 'filled', 'MarkerFaceAlpha', 0.45, 'MarkerFaceColor', [0.2 0.45 0.75]);
hold on;
plot(mean(mcENU(:,1)) + ell(:,1), mean(mcENU(:,2)) + ell(:,2), 'r-', 'LineWidth', 1.8);
plot(0, 0, 'k+', 'MarkerSize', 14, 'LineWidth', 1.6);
set(gca, 'Color','w', 'XColor','k', 'YColor','k'); axis equal; grid on;
xlabel('East error [m]', 'Color','k'); ylabel('North error [m]', 'Color','k');
title(sprintf('Horizontal scatter of the MC solutions, \\sigma_t = 10^{-8} s (95%% ellipse semi-axis = %.2f m)', semiMajor95), 'Color','k');
legend({'Monte Carlo solutions', '95% confidence ellipse', 'reference solution'}, 'TextColor','k', 'Location','best');
drawnow;
exportgraphics(fig, fullfile(outDir,'mc_error_ellipse.png'), 'Resolution',150, 'BackgroundColor','white');

fprintf('\nHorizontal error statistics for sigma_t = 1e-8 s:\n');
fprintf('  std East = %.3f m, std North = %.3f m, std Up = %.3f m\n', ...
    std(mcENU(:,1)), std(mcENU(:,2)), std(mcENU(:,3)));
fprintf('  semi-major axis of the 95%% confidence ellipse = %.3f m\n\n', semiMajor95);


%% 6d) SENSITIVITY: NOISE IN THE SATELLITE POSITIONS
fprintf('------------------------------------------------------------\n');
fprintf(' 6d) EFFECT OF NOISE IN SATELLITE POSITIONS (simulated ephemeris error)\n');
fprintf('------------------------------------------------------------\n');

sigma_pos_list = [0, 1, 10, 50, 100];   % meters, added to satellite X, Y, Z
errMeanPos = zeros(size(sigma_pos_list));
errStdPos  = zeros(size(sigma_pos_list));

fprintf('%14s | %10s %10s\n', 'sigma_pos [m]', 'mean err[m]', 'std err[m]');
for k = 1:numel(sigma_pos_list)
    sp   = sigma_pos_list(k);
    errs = zeros(nMC,1);
    for m = 1:nMC
        Sat_noisy = Sat + sp*randn(N,3);
        pk = lsqnonlin(@(p) gps_model(p, Sat_noisy, d), x0_base(:), [], [], optsLsqQuiet);
        errs(m) = norm(pk(:) - p_ref(:));
    end
    errMeanPos(k) = mean(errs);
    errStdPos(k)  = std(errs);
    fprintf('%14.1f | %10.3f %10.3f\n', sp, errMeanPos(k), errStdPos(k));
end

fig = figure('Visible','off', 'Color','w', 'Position',[100 100 900 550]);
errorbar(sigma_pos_list, errMeanPos, errStdPos, '-s', 'LineWidth', 1.5, 'MarkerSize', 6, 'Color', [0.85 0.325 0.098]);
set(gca, 'Color','w', 'XColor','k', 'YColor','k'); grid on;
xlabel('Satellite position noise std. dev. \sigma_{pos} [m]', 'Color','k');
ylabel('Position error vs. reference solution [m]', 'Color','k');
title('Effect of satellite position noise on the estimated position (MC, n=200)', 'Color','k');
drawnow;
exportgraphics(fig, fullfile(outDir,'satellite_noise_sensitivity.png'), 'Resolution',150, 'BackgroundColor','white');


%% Save results
save(fullfile(outDir,'results.mat'), 'Sat', 'd', 'p_ref', 'p_own', 'lat_sol', 'lon_sol', 'h_sol', ...
    'elev', 'azim', 'dop', 'lsqHistory', 'tols', 'tolIter', 'tolErr', ...
    'sigma_t_list', 'errMean', 'errStd', 'errMax', 'mcENU', ...
    'sigma_pos_list', 'errMeanPos', 'errStdPos', 'errGrid', 'radii', 'angles');

fprintf('\nFigures and results written to: %s\n', outDir);

diary off;
