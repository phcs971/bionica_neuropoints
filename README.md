```mermaid
flowchart TD
    main(Main)
    subgraph Helpers
        plat(Platform)
        screen(Screen Brightness)
    end
    subgraph Models

    end
    main --> Helpers
    main & Helpers --> Models
```