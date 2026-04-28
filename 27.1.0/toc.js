// Populate the sidebar
//
// This is a script, and not included directly in the page, to control the total size of the book.
// The TOC contains an entry for each page, so if each page includes a copy of the TOC,
// the total size of the page becomes O(n**2).
class MDBookSidebarScrollbox extends HTMLElement {
    constructor() {
        super();
    }
    connectedCallback() {
        this.innerHTML = '<ol class="chapter"><li class="chapter-item expanded affix "><a href="introduction.html">Introduction</a></li><li class="chapter-item expanded "><a href="installation.html"><strong aria-hidden="true">1.</strong> Installation</a></li><li class="chapter-item expanded "><a href="key-features.html"><strong aria-hidden="true">2.</strong> Key Features</a></li><li class="chapter-item expanded "><a href="getting-started.html"><strong aria-hidden="true">3.</strong> Getting Started</a></li><li class="chapter-item expanded "><a href="vollo-compiler.html"><strong aria-hidden="true">4.</strong> Vollo Compiler</a></li><li><ol class="section"><li class="chapter-item expanded "><a href="api.html"><strong aria-hidden="true">4.1.</strong> API Reference</a></li><li class="chapter-item expanded "><a href="supported-models.html"><strong aria-hidden="true">4.2.</strong> Supported Models</a></li><li class="chapter-item expanded "><a href="example-1-mlp.html"><strong aria-hidden="true">4.3.</strong> Example 1: MLP</a></li><li class="chapter-item expanded "><a href="example-2-cnn.html"><strong aria-hidden="true">4.4.</strong> Example 2: CNN</a></li><li class="chapter-item expanded "><a href="example-3-lstm.html"><strong aria-hidden="true">4.5.</strong> Example 3: LSTM</a></li><li class="chapter-item expanded "><a href="example-4-mixed-precision.html"><strong aria-hidden="true">4.6.</strong> Example 4: Mixed Precision</a></li><li class="chapter-item expanded "><a href="example-5-multi-model.html"><strong aria-hidden="true">4.7.</strong> Example 5: Multiple Models in a Vollo Program</a></li><li class="chapter-item expanded "><a href="data-dimension.html"><strong aria-hidden="true">4.8.</strong> The data dimension</a></li><li class="chapter-item expanded "><a href="vollo-onnx.html"><strong aria-hidden="true">4.9.</strong> ONNX Support</a></li></ol></li><li class="chapter-item expanded "><a href="benchmark.html"><strong aria-hidden="true">5.</strong> Benchmarks</a></li><li><ol class="section"><li class="chapter-item expanded "><a href="benchmark-mlp.html"><strong aria-hidden="true">5.1.</strong> MLP</a></li><li class="chapter-item expanded "><a href="benchmark-cnn.html"><strong aria-hidden="true">5.2.</strong> CNN</a></li><li class="chapter-item expanded "><a href="benchmark-lstm.html"><strong aria-hidden="true">5.3.</strong> LSTM</a></li><li class="chapter-item expanded "><a href="benchmark-io.html"><strong aria-hidden="true">5.4.</strong> IO Round Trip</a></li></ol></li><li class="chapter-item expanded "><a href="accelerator-setup.html"><strong aria-hidden="true">6.</strong> Accelerator Setup</a></li><li><ol class="section"><li class="chapter-item expanded "><a href="system-requirements.html"><strong aria-hidden="true">6.1.</strong> System Requirements</a></li><li class="chapter-item expanded "><a href="programming-the-agilex.html"><strong aria-hidden="true">6.2.</strong> Programming the Agilex</a></li><li class="chapter-item expanded "><a href="programming-the-artena.html"><strong aria-hidden="true">6.3.</strong> Programming the Artena</a></li><li class="chapter-item expanded "><a href="programming-the-v80.html"><strong aria-hidden="true">6.4.</strong> Programming the V80</a></li><li class="chapter-item expanded "><a href="troubleshooting-the-v80.html"><strong aria-hidden="true">6.5.</strong> Troubleshooting the V80</a></li><li class="chapter-item expanded "><a href="licensing.html"><strong aria-hidden="true">6.6.</strong> Licensing</a></li><li class="chapter-item expanded "><a href="running-an-example.html"><strong aria-hidden="true">6.7.</strong> Running an Example</a></li><li class="chapter-item expanded "><a href="running-the-benchmark.html"><strong aria-hidden="true">6.8.</strong> Running the Benchmark</a></li></ol></li><li class="chapter-item expanded "><a href="vollo-runtime.html"><strong aria-hidden="true">7.</strong> Vollo Runtime</a></li><li><ol class="section"><li class="chapter-item expanded "><a href="c-api.html"><strong aria-hidden="true">7.1.</strong> C API</a></li><li class="chapter-item expanded "><a href="vollo-rt-example.html"><strong aria-hidden="true">7.2.</strong> C Example</a></li><li class="chapter-item expanded "><a href="vollo-rt-python-example.html"><strong aria-hidden="true">7.3.</strong> Python Example</a></li></ol></li><li class="chapter-item expanded "><a href="ip-core/0-intro.html"><strong aria-hidden="true">8.</strong> Vollo IP Core</a></li><li><ol class="section"><li class="chapter-item expanded "><a href="ip-core/1-selecting-an-ip-core.html"><strong aria-hidden="true">8.1.</strong> Selecting an IP Core</a></li><li class="chapter-item expanded "><a href="ip-core/2-interface.html"><strong aria-hidden="true">8.2.</strong> IP Core Interface</a></li><li class="chapter-item expanded "><a href="ip-core/3-quartus-integration.html"><strong aria-hidden="true">8.3.</strong> Quartus Integration</a></li><li class="chapter-item expanded "><a href="ip-core/4-config.html"><strong aria-hidden="true">8.4.</strong> Runtime configuration</a></li><li class="chapter-item expanded "><a href="ip-core/5-example-design.html"><strong aria-hidden="true">8.5.</strong> Example design</a></li></ol></li><li class="chapter-item expanded "><a href="versions.html"><strong aria-hidden="true">9.</strong> Versions</a></li><li><ol class="section"><li class="chapter-item expanded "><a href="release-notes.html"><strong aria-hidden="true">9.1.</strong> Release Notes</a></li></ol></li></ol>';
        // Set the current, active page, and reveal it if it's hidden
        let current_page = document.location.href.toString().split("#")[0].split("?")[0];
        if (current_page.endsWith("/")) {
            current_page += "index.html";
        }
        var links = Array.prototype.slice.call(this.querySelectorAll("a"));
        var l = links.length;
        for (var i = 0; i < l; ++i) {
            var link = links[i];
            var href = link.getAttribute("href");
            if (href && !href.startsWith("#") && !/^(?:[a-z+]+:)?\/\//.test(href)) {
                link.href = path_to_root + href;
            }
            // The "index" page is supposed to alias the first chapter in the book.
            if (link.href === current_page || (i === 0 && path_to_root === "" && current_page.endsWith("/index.html"))) {
                link.classList.add("active");
                var parent = link.parentElement;
                if (parent && parent.classList.contains("chapter-item")) {
                    parent.classList.add("expanded");
                }
                while (parent) {
                    if (parent.tagName === "LI" && parent.previousElementSibling) {
                        if (parent.previousElementSibling.classList.contains("chapter-item")) {
                            parent.previousElementSibling.classList.add("expanded");
                        }
                    }
                    parent = parent.parentElement;
                }
            }
        }
        // Track and set sidebar scroll position
        this.addEventListener('click', function(e) {
            if (e.target.tagName === 'A') {
                sessionStorage.setItem('sidebar-scroll', this.scrollTop);
            }
        }, { passive: true });
        var sidebarScrollTop = sessionStorage.getItem('sidebar-scroll');
        sessionStorage.removeItem('sidebar-scroll');
        if (sidebarScrollTop) {
            // preserve sidebar scroll position when navigating via links within sidebar
            this.scrollTop = sidebarScrollTop;
        } else {
            // scroll sidebar to current active section when navigating via "next/previous chapter" buttons
            var activeSection = document.querySelector('#sidebar .active');
            if (activeSection) {
                activeSection.scrollIntoView({ block: 'center' });
            }
        }
        // Toggle buttons
        var sidebarAnchorToggles = document.querySelectorAll('#sidebar a.toggle');
        function toggleSection(ev) {
            ev.currentTarget.parentElement.classList.toggle('expanded');
        }
        Array.from(sidebarAnchorToggles).forEach(function (el) {
            el.addEventListener('click', toggleSection);
        });
    }
}
window.customElements.define("mdbook-sidebar-scrollbox", MDBookSidebarScrollbox);
