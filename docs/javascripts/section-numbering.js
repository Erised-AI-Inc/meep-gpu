(function () {
  const NUMBER_CLASS = "manual-section-number";
  const HEADING_CLASS = "manual-heading-number";

  function clean(root) {
    root.querySelectorAll("." + NUMBER_CLASS + ", ." + HEADING_CLASS).forEach((node) => node.remove());
  }

  function insert(target, value, className) {
    const number = document.createElement("span");
    number.className = className;
    number.textContent = value;
    number.setAttribute("aria-hidden", "true");
    target.insertBefore(number, target.firstChild);
  }

  function labels(item) {
    return Array.from(item.querySelectorAll(":scope > a.md-nav__link > .md-ellipsis, :scope > label.md-nav__link > .md-ellipsis"));
  }

  function numberNav(list, prefix, level) {
    let index = 0;
    Array.from(list.children).filter((item) => item.classList && item.classList.contains("md-nav__item")).forEach((item) => {
      const items = labels(item);
      const title = (items[0] && items[0].textContent || "").trim();
      const skip = level === 0 && title === "Home";
      let number = prefix;
      if (!skip && items.length) {
        index += 1;
        number = prefix ? prefix + "." + index : String(index);
        items.forEach((label) => insert(label, number, NUMBER_CLASS));
      }
      const nested = item.querySelector(":scope > nav.md-nav > ul.md-nav__list");
      if (nested) numberNav(nested, skip ? prefix : number, level + 1);
    });
  }

  function pageNumber() {
    const path = new URL(window.location.href).pathname.replace(/\/index\.html$/, "/").replace(/\/$/, "");
    const link = Array.from(document.querySelectorAll(".md-nav--primary a.md-nav__link[href]")).find((candidate) => {
      return new URL(candidate.href, window.location.href).pathname.replace(/\/index\.html$/, "/").replace(/\/$/, "") === path;
    });
    return link && link.querySelector("." + NUMBER_CLASS) ? link.querySelector("." + NUMBER_CLASS).textContent : "";
  }

  function numberHeadings(root, base) {
    const article = root.querySelector(".md-content__inner.md-typeset");
    if (!article || !base) return;
    const h1 = article.querySelector("h1");
    if (h1) insert(h1, base, HEADING_CLASS);
    let h2 = 0;
    let h3 = 0;
    article.querySelectorAll("h2, h3").forEach((heading) => {
      if (heading.tagName === "H2") {
        h2 += 1;
        h3 = 0;
        insert(heading, base + "." + h2, HEADING_CLASS);
      } else if (h2) {
        h3 += 1;
        insert(heading, base + "." + h2 + "." + h3, HEADING_CLASS);
      }
    });
  }

  function bind(root = document) {
    clean(root);
    const nav = document.querySelector(".md-nav--primary > .md-nav__list");
    if (!nav) return;
    numberNav(nav, "", 0);
    numberHeadings(document, pageNumber());
  }

  document.addEventListener("DOMContentLoaded", () => bind(document));
  if (typeof document$ !== "undefined" && document$ && typeof document$.subscribe === "function") {
    document$.subscribe(() => bind(document));
  }
}());
