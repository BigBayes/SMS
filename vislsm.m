% Visualize the convergence of local sample means.
% 
% inputs:
%   lsm: a 3D array of size d x (m+1) x niters, holding local sample means 
%   during iterations. The "+1" dimension corresponds to the mean of the
%   approximate posterior. It is one of the outputs of smssample.
% 
%   tm: the true posterior mean (d x 1) as computed from the ground truth
%   samples.
% 
%   ii (optional): a subset of 1:d indicating the dimensions to plot.
%   (plotting all dimensions by default)
% 
%   doshift (optional): a boolean scalar indicating whether to shift the
%   curves so that all of them converge to zero in the end. (false by
%   default)
% 
% Minjie Xu (chokkyvista06@gmail.com)

function vislsm(lsm, tm, ii, doshift)
if ~exist('ii', 'var')
    ii = 1:size(lsm,1);
end
if ~exist('doshift', 'var')
    doshift = false;
end
mkrs = {'.'};%{'o','s','d','^','v','p','h','x'};
clrs = {'r','g','b','c','m','y','k'};
mi = 0;

niters = size(lsm, 3);
figure; hold on;
for i = ii
    [mkr, clr] = nxtmkrclr;
    if doshift
        dv = lsm(i, end, end);
    else
        dv = 0;
        if exist('tm', 'var')
            plot([1,niters], [tm(i), tm(i)], clr, 'LineWidth', 1);
        end
    end
    plot(squeeze(lsm(i, 1:end-1, :))'-dv, ['-.',clr,mkr], ...
        'MarkerSize', 10, 'LineWidth', 1.5);
    plot(squeeze(lsm(i, end, :))'-dv, [clr,'*'], ...
        'MarkerSize', 10, 'LineWidth', 2);
end
set(gca, 'XTick', 1:niters);
ylims = get(gca, 'YLim');
for i = 2:niters
    plot([i,i], ylims, 'k:');
end
hold off;
xlabel('# iterations');

    function [mkr, clr] = nxtmkrclr
        mkr = mkrs{mod(mi, numel(mkrs)) + 1};
        clr = clrs{mod(mi, numel(clrs)) + 1};
        mi = mi + 1;
    end

end
