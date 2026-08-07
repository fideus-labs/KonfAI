// Mermaid stamps width="100%" on the <svg>, and the extension's stylesheet pins a
// 500px height, so a narrow diagram is stretched to the column and a tall one is
// squashed. Natural size is the ceiling here: shrink to fit, never stretch.
//
// The label font is left alone on purpose. Mermaid measures every box with its own
// font before drawing, so swapping the family afterwards clips the text.
(function () {
    function fit(svg) {
        const box = svg.getAttribute("viewBox");
        if (!box) return;
        const width = parseFloat(box.split(/[\s,]+/)[2]);
        if (!width) return;
        svg.removeAttribute("width");
        svg.removeAttribute("height");
        svg.style.width = width + "px";
        svg.style.maxWidth = "100%";
        svg.style.height = "auto";
    }

    function fitAll() {
        document.querySelectorAll(".mermaid svg, pre.mermaid svg").forEach(fit);
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", fitAll);
    } else {
        fitAll();
    }
    window.addEventListener("load", () => {
        fitAll();
        setTimeout(fitAll, 300);
    });
    new MutationObserver(fitAll).observe(document.documentElement, { childList: true, subtree: true });
})();
