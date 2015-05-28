% Plot the errors (as computed by 'errmsr') of different methods.
% 
% inputs
%   mm: the row vector of "# of partitions".
% 
%   ni: indices of the multiple runs to show, a subset of 1:nrun.
% 
%   mi: indices of the multiple partition schemes to show, a subset of
%   1:numel(mm).
% 
%   errs: a cell array, each entry of which is the 'errs' (a nrun*numel(mm)
%   cell array in itself) returned by the corresponding method.
% 
%   labels: a cell array of string labels for each method.
% 
%   logscale (optional): boolean, whether to use log scale or linear scale
%   (default) for the y-axis.
% 
%   xfdname (optional): decides the meaning of the x-axis, 'nssamples' or 
%   npsamples'.
%   'nssamples' means the total # of samples drawn so far overall;
%   'npsamples' means the total # of samples drawn so far locally on each
%   node.
% 
%   mkr (optional): a cell array of marker types to use.
% 
%   figdir (optional): when available, where the generated figures will be
%   saved.
% 
%   fdnames (optional): a cell array of the subset of 'errs' to plot.
% 
% examples:
%   load('data.mat', 'nrun', 'mm');
%   SMSs = load('dpep_sync.mat', 'errs');
%   SMSa = load('dpep_async.mat', 'errs');
%   XINGp = load('xing_p.mat', 'errs');
%   XINGn = load('xing_n.mat', 'errs');
%   WEIS = load('weis.mat', 'errs');
% 
%   viserrs(mm, 1, 1:numel(mm), {SMSs.errs, SMSa.errs}, {'SMS(s)', 'SMS(a)'}, 'nssamples');
%   viserrs(mm, 1:nrun, 1, {SMSs.errs, XINGp.errs, XINGn.errs, WEIS.errs}, {'SMS(s)', 'XING(p)', 'XING(n)', 'WEIS'});
% 
% Minjie Xu (chokkyvista06@gmail.com)

function viserrs(mm, ni, mi, errs, labels, logscale, xfdname, mkr, figdir, fdnames, N)
dpsi = find(cellfun(@(c)~isempty(c), strfind(labels, 'SMS(s')), 1);
dpai = find(cellfun(@(c)~isempty(c), strfind(labels, 'SMS(a')), 1);
if ~exist('logscale', 'var') || isempty(logscale) || ~logscale
    hplot = @plot;
else
    hplot = @semilogy;
end
if ~exist('xfdname', 'var') || isempty(xfdname)
    if ~isempty(dpai)
        xfdname = 'nssamples';
    else
        xfdname = 'npsamples';
    end
end
if ~exist('mkr', 'var') || isempty(mkr)
    mkr = {'x','o','s','d','^','v','p','h','x'};
end
if ~exist('fdnames', 'var') || isempty(fdnames)
    fdnames = fieldnames(errs{1}{1});
end
H = figure;
cc = lines(7);
for i = 1:numel(fdnames)
    yfdname = fdnames{i};
    if any(strcmp(yfdname, {'npsamples', 'nssamples'}))
        continue;
    end
    clf(H, 'reset');
    legl = zeros(1,numel(errs));
    legm = zeros(1,numel(mi));
    for j = 1:numel(errs)
        cerr = errs{j};
        if ~isfield(cerr{1}, yfdname)
            continue;
        end
        for k = 1:numel(mi)
            for l = ni
                if isempty(cerr{l,mi(k)})
                    continue;
                end
                nsamples = cumsum([cerr{l,mi(k)}.(xfdname)]);
                yvals = [cerr{l,mi(k)}.(yfdname)];
                xvals = nsamples;
                if exist('N', 'var')
                    xvals = nsamples*round(N/mi(k));
                end
                if legl(j) == 0
                    if numel(mi)==1
                        legl(j) = hplot(xvals(1), yvals(1), ...
                            ['--',mkr{mod(j-1,numel(mkr))+1}], 'color', cc(mod(j-1,size(cc,1))+1,:), ...
                            'MarkerSize', 10, 'LineWidth', 2);
                    else 
                        legl(j) = hplot(xvals(1), yvals(1), ...
                            mkr{mod(j-1,numel(mkr))+1}, 'MarkerSize', 10);
                    end
                    hold on;
                end
                if legm(k) == 0
                    legm(k) = hplot(xvals(1), yvals(1), ...
                        '--', 'color', cc(mod(k-1,size(cc,1))+1,:), 'LineWidth', 2);
                    hold on;
                end
                if numel(mi)==1
                    hplot(xvals, yvals, ...
                        ['--',mkr{mod(j-1,numel(mkr))+1}], 'color', cc(mod(j-1,size(cc,1))+1,:), ...
                        'MarkerSize', 10, 'LineWidth', 2);
                else
                    hplot(xvals, yvals, ...
                        ['--',mkr{mod(j-1,numel(mkr))+1}], 'color', cc(mod(k-1,size(cc,1))+1,:), ...
                        'MarkerSize', 10, 'LineWidth', 2);
                end
                hold on;
            end
        end
    end
    if numel(mi) == 1
        legend(nonzeros(legl), labels(legl~=0));
        title(num2str(mm(mi), 'm = %d'));
        nsamples = [];
        if ~isempty(dpsi)
            nsamples = cumsum([errs{dpsi}{ni(1),mi}.(xfdname)]);
        elseif ~isempty(dpai)
            if strcmp(xfdname, 'nssamples')
                terr = errs{dpai}{ni(1),mi};
                T = terr(1).npsamples;
                nsamples = T*mm(mi):T*mm(mi):sum([terr.nssamples]);
            else
                nsamples = [];
            end
        end
        if ~isempty(nsamples)
            set(gca, 'XTick', nsamples);
            ylims = get(gca, 'YLim');
            for ns = nsamples
                hplot([ns,ns], ylims, 'k:');
            end
            xlim([0,nsamples(end)]);
        end
    else
        legend([legl,legm], [labels,cellstr(num2str(mm(mi)', 'm = %d'))']);
    end
    ylabel(yfdname, 'Interpreter', 'none');
%     figfont;
    pause;
    if exist('figdir', 'var')
        savecurfig(figdir, yfdname);
    end
end
close(H);

end